from sanic import Sanic
import reactpy
from reactpy.backend.sanic import configure
import subprocess
from sanic.log import logger
from sanic import Request
from sanic import json
import multiprocessing as mp
import os
from src.sanic_vec_env import SanicVecEnv
from src.trader import Trader
from src.ingestion import CVIngestionPipeline
from functools import partial
from stable_baselines3 import SAC as model
from stable_baselines3.common.vec_env import VecMonitor
import os

app = Sanic(__name__)
app_path = os.path.dirname(__file__)
app.static("/static", os.path.join(app_path, "static"))
log_dir = os.path.join(app_path, "log")

@app.post("/start")
async def start(request: Request):
    # args = request.json

    return json({"status": "success"})


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
        env = SanicVecEnv([partial(Trader, train, test=True)
                          for _ in range(16)], app)
        env = VecMonitor(env, log_dir)
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
            batch_size=128,
            use_sde=True,
        )
        model_train.learn(total_timesteps=int(500), progress_bar=True)
        model_train.save(f"model_{i}")
        env.close()
    return json({"status": "success"})


@reactpy.component
def Button():
    logger.info("Rendering button")
    return reactpy.html.button({"on_click": start_handler}, "Click me!")


@app.main_process_start
async def main_process_start(app):
    app.shared_ctx.mp_ctx = mp.get_context("spawn")


@app.route("/stop", methods=["POST"])
async def stop(request: Request):
    data = request.json
    model_file_handle = data.get("model_file_handle")
    os.system(
        f"kill -9 $(ps aux | grep {model_file_handle} | awk '{{print $2}}')")
    return json({"status": "success"})


@app.route("/status", methods=["POST"])
async def status(request: Request):
    data = request.json
    model_file_handle = data.get("model_file_handle")
    output = subprocess.check_output(
        f"ps aux | grep {model_file_handle}", shell=True)
    return json({"status": output})


@reactpy.component
def ReactApp():
    return reactpy.html.div({}, Button())


configure(app, ReactApp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, workers=16)
