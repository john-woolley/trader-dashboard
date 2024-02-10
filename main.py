from sanic import Sanic
import reactpy
import sanic
from reactpy.backend.sanic import configure
from sanic.log import logger
from sanic import Request
from sanic import json, redirect
import multiprocessing as mp
import os
from src.sanic_vec_env import SanicVecEnv
from src.trader import Trader
from src.ingestion import CVIngestionPipeline
from stable_baselines3 import SAC as model
from stable_baselines3.common.vec_env import VecMonitor
import src.db as db
import numpy as np
import pandas as pd

CONN = "postgresql+psycopg2://trader_dashboard@0.0.0.0:5432/trader_dashboard"
app = Sanic(__name__)
app_path = os.path.dirname(__file__)
app.static("/static", os.path.join(app_path, "static"))
log_dir = os.path.join(app_path, "log")


@app.main_process_start
async def main_process_start(app):
    logger.info(
        f"Created server process running Sanic Version {sanic.__version__}")
    logger.debug("Main process started")
    app.shared_ctx.mp_ctx = mp.get_context("spawn")
    logger.debug("Created shared context")
    app.shared_ctx.hallpass = mp.Semaphore()
    logger.debug("Created hallpass")
    db.drop_workers_table()
    logger.debug("Dropped old workers table")
    db.drop_jobs_table()
    logger.debug("Dropped old jobs table")
    db.create_jobs_table()
    logger.debug("Created new jobs table")
    db.create_workers_table()
    logger.debug("Created new workers table")


@app.get("/upload_csv")
def upload_csv(request: Request):
    input_path = request.args.get("file")
    output_path = request.args.get("output")
    with open(input_path, "r") as f:
        input_file = f.read()
    if not output_path:
        output_path = input_path.split("/")[-1].split(".")[0]
    file_path = os.path.join(app_path, 'data', output_path)
    logger.info(f"Uploading {input_path} to {file_path}")
    with open(file_path, "w") as f:
        f.write(input_file)
    logger.info(f"Uploaded {input_path} to {file_path}")
    df = pd.read_csv(file_path, parse_dates=True)
    db.insert_raw_table(df, output_path)
    return json({"status": "success"})


@app.get("/start")
def start_handler(request: Request):

    logger.info("Starting training")
    cv_periods = int(request.args.get("cv_periods", 1))
    train_start_date = request.args.get("train_start_date")
    ncpu = int(request.args.get("ncpu", 1))
    jobname = request.args.get("jobname", "default")
    render_mode = request.args.get("render_mode", "none")
    db.add_job(jobname)
    timesteps = int(request.args.get("timesteps", 1000))
    logger.info(
        f"Starting training with cv_periods={cv_periods}"
        f"and train_start_date={train_start_date}")
    file_path = os.path.join(app_path, "data/master.csv")
    data = CVIngestionPipeline(
        file_path, cv_periods, start_date=train_start_date)
    for i in range(0, len(data)):
        logger.info(f"Training on fold {i+1} of {len(data)}")
        train, test = tuple(*iter(data))
        def env_fn(): return Trader(train, test=True, render_mode=render_mode)
        app.shared_ctx.hallpass.acquire()
        logger.info("Acquired hallpass")
        try:
            env = SanicVecEnv([env_fn for _ in range(ncpu)],
                              request.app, jobname)
            env = VecMonitor(env, log_dir)
        except Exception as e:
            logger.error("Error creating environment")
            logger.error(e)
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
        model_train.learn(total_timesteps=timesteps)
        model_train.save(f"model_{i}")
        env_test = Trader(test, test=True)
        model_handle = f"model_{i}"
        model_test = model.load(model_handle, env=env_test)
        vec_env = model_test.get_env()
        obs = vec_env.reset()
        lstm_states = None
        num_envs = 1
        episode_starts = np.ones((num_envs,), dtype=bool)
        for j in range(len(test["close"]) - 1):
            action, lstm_states = model_test.predict(
                obs, state=lstm_states, episode_start=episode_starts
            )
            obs, reward, done, info = vec_env.step(action)
            if done:
                break
        test_render_handle = f"test_render_{i}.csv"
        env_test.render_df.to_csv(test_render_handle)
    return json({"status": "success", "model_name": model_handle, "test_render": test_render_handle})


@app.get("/stop")
def stop_handler(request: Request):
    jobname = request.args.get("jobname", "default")
    logger.info(f"Stopping training for job: {jobname}")
    request.app.shared_ctx.hallpass.acquire()
    logger.info("Acquired hallpass")
    try:
        workers = db.get_workers_by_name(jobname)
        request.app.m.terminate_worker(workers)
    except Exception as e:
        logger.error("Error stopping workers")
        logger.error(e)
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
