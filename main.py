from sanic import Sanic
import reactpy
import sanic
from reactpy.backend.sanic import configure
from sanic.log import logger
from sanic import Request
from sanic import json
import multiprocessing as mp
import os
from src.sanic_vec_env import SanicVecEnv
from src.trader import Trader
from src.ingestion import CVIngestionPipeline
from stable_baselines3 import SAC as model
from stable_baselines3.common.vec_env import VecMonitor
import os
import psycopg2
import sqlalchemy as sa
import psutil

def getconn():
    c = psycopg2.connect(user="trader_dashboard", host="0.0.0.0", dbname="trader_dashboard")
    return c 

app = Sanic(__name__)

app_path = os.path.dirname(__file__)
app.static("/static", os.path.join(app_path, "static"))
log_dir = os.path.join(app_path, "log")

@app.main_process_start
async def main_process_start(app):
    logger.info(f"Created server process running Sanic Version {sanic.__version__}")
    logger.debug("Main process started")
    app.shared_ctx.mp_ctx = mp.get_context("spawn")
    logger.debug("Created shared context")
    app.shared_ctx.hallpass = mp.Semaphore()
    logger.debug("Created hallpass")


@app.get("/start")
def start_handler(request: Request):

    logger.info("Starting training")
    args = {"cv_periods": 5, "train_start_date": "2019-04-03"}
    cv_periods = args.get("cv_periods", 1)
    train_start_date = args.get("train_start_date")
    logger.info(
        f"Starting training with cv_periods={cv_periods} and train_start_date={train_start_date}")
    file_path = os.path.join(app_path, "data/master.csv")
    data = CVIngestionPipeline(file_path, cv_periods, start_date=train_start_date)
    for i in range(0, len(data)):
        logger.info(f"Training on fold {i+1} of {len(data)}")
        train, test = tuple(*iter(data))
        env_fn = lambda: Trader(train, test=True)
        app.shared_ctx.hallpass.acquire()
        logger.info("Acquired hallpass")
        try:
            env = SanicVecEnv([env_fn for _ in range(2)], request.app)
            env = VecMonitor(env, log_dir)
        except Exception as e:
            logger.info("Error creating environment")
            logger.info(e)
            return json({"status": "error"})
        finally:
            app.shared_ctx.hallpass.release()
            logger.info("Released hallpass")
        policy_kwargs = dict(
            net_arch=dict(
                pi=[4096, 2048, 1024, 1024], 
                vf=[4096, 2048, 1024, 1024],
                qf=[4096, 2048, 1024, 1024]
                )
        )
        model_train = model(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            batch_size=1,
            use_sde=True,
        )
        model_train.learn(total_timesteps=int(1e6), progress_bar=True)
        model_train.save(f"model_{i}")
    return json({"status": "success"})


@app.get("/stop")
def stop_handler(request: Request):
    logger.info("Stopping training")
    request.app.shared_ctx.hallpass.acquire()
    logger.info("Acquired hallpass")
    try:
        pool = sa.pool.QueuePool(getconn, max_overflow=0, pool_size=24)
        conn = sa.create_engine('postgresql://trader_dashboard/trader_dashboard', pool=pool).connect()
        worker_query = sa.select("*").select_from(sa.table("workers"))
        workers = list(map(lambda x: x[1], conn.execute(worker_query).fetchall()))
        for worker in workers:
            request.app.m.restart(worker)
    except Exception as e:
        logger.info("Error stopping workers")
        logger.info(e)
        return json({"status": "error"})
    finally:
        request.app.shared_ctx.hallpass.release()
        logger.info("Released hallpass")
    return json({"status": "success"})

@reactpy.component
def Button():
    logger.info("Rendering button")
    return reactpy.html.button({"on_click": start_handler}, "Click me!")

@reactpy.component
def ReactApp():
    return reactpy.html.div({}, Button())


configure(app, ReactApp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, workers=16)
