# Description: Main file for the trader dashboard
"""
This file contains the main code for the trader dashboard application.
It sets up the Sanic server, handles requests for uploading CSV files,
starting training, and stopping training. 
It also includes functions for creating the React components used in
the application.
"""
import multiprocessing as mp
import os

from functools import partial

import reactpy
import sanic
import numpy as np
import pandas as pd

from stable_baselines3 import SAC as model
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from reactpy.backend.sanic import configure
from sanic.log import logger
from sanic import Request, json, Sanic

from src import db
from src.sanic_vec_env import SanicVecEnv
from src.trader import Trader
from src.ingestion import CVIngestionPipeline

CONN = "postgresql+psycopg2://trader_dashboard@0.0.0.0:5432/trader_dashboard"
main_app = Sanic(__name__)
app_path = os.path.dirname(__file__)
main_app.static("/static", os.path.join(app_path, "static"))
log_dir = os.path.join(app_path, "log")


@main_app.main_process_start
async def main_process_start(app):
    """
    Starts the main process of the trader dashboard application.

    Args:
        app: The Sanic application object.

    Returns:
        None
    """
    sanic_ver = sanic.__version__
    logger.info("Created server process running Sanic Version %s", sanic_ver)
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


@main_app.get("/upload_csv")
def upload_csv(request: Request):
    """
    Uploads a CSV file, saves it to a specified output path,
    and inserts its contents into a database as a raw table.

    Args:
        request (Request): The request object containing the file path,
        output path, parse_dates flag, and index_col.

    Returns:
        dict: A dictionary with the status of the upload process.
    """

    input_path = request.args.get("file")
    output_path = request.args.get("output")
    parse_dates = bool(int(request.args.get("parse_dates", 0)))
    index_col = request.args.get("index_col")
    with open(input_path, "r", encoding='utf-8') as f:
        input_file = f.read()
    if not output_path:
        output_path = input_path.split("/")[-1].split(".")[0]
    file_path = os.path.join(app_path, "data", output_path)
    logger.info("Uploading %s to %s", input_path, file_path)
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(input_file)
    logger.info("Uploaded %s to %s", input_path, file_path)
    df = pd.read_csv(file_path, parse_dates=parse_dates, index_col=index_col)
    db.insert_raw_table(df, output_path)
    logger.info("Inserted %s into database as raw table", output_path)
    return json({"status": "success"})


@main_app.get("/start")
def start_handler(request: Request):
    """
    Handles the start request for training.

    Args:
        request (Request): The request object containing the run parameters.

    Returns:
        JSON response indicating the status of the training process,
        the model name, and the test render file name.
    """

    logger.info("Starting training")
    cv_periods = int(request.args.get("cv_periods", 1))
    start_date = request.args.get("train_start_date")
    ncpu = int(request.args.get("ncpu", 1))
    jobname = request.args.get("jobname", "default")
    render_mode = request.args.get("render_mode", "none")
    db.add_job(jobname)
    timesteps = int(request.args.get("timesteps", 1000))
    logger.info(
        "Starting training with cv_periods=%s and train_start_date=%s",
        cv_periods, start_date
    )
    file_path = os.path.join(app_path, "data/master.csv")
    data = CVIngestionPipeline(file_path, cv_periods, start_date=start_date)
    for i in range(0, len(data)):
        logger.info("Training on fold %i of %i", i, len(data))
        train, test = tuple(*iter(data))

        env_fn = partial(Trader, train, test=True, render_mode=render_mode)
        request.app.shared_ctx.hallpass.acquire()
        logger.info("Acquired hallpass")
        try:
            env_fns = [env_fn for _ in range(ncpu)]
            env = SanicVecEnv(env_fns, request.app, jobname)  # type: ignore
            env = VecMonitor(env, log_dir)
        except MemoryError as e:
            logger.error("Out of memory error creating environment")
            logger.error(e)
            return json({"status": "error"})
        finally:
            request.app.shared_ctx.hallpass.release()
            logger.info("Released hallpass")
        policy_kwargs = {
            "net_arch": {
                "pi": [4096, 2048, 1024, 1024],
                "vf": [4096, 2048, 1024, 1024],
                "qf": [4096, 2048, 1024, 1024],
            }
        }
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
        try:
            assert isinstance(vec_env, DummyVecEnv)
        except AssertionError:
            logger.error("Error loading environment")
            return json({"status": "error"})
        obs = vec_env.reset()
        lstm_states = None
        num_envs = 1
        episode_starts = np.ones((num_envs,), dtype=bool)
        for _ in range(len(test["close"]) - 1):
            action, lstm_states = model_test.predict(
                #  We ignore the type error here because the type hint for obs
                #  is not correct
                obs,  # type: ignore
                state=lstm_states,
                episode_start=episode_starts
            )
            obs, _, done, _ = vec_env.step(action)
            if done:
                break
        test_render_handle = f"test_render_{i}.csv"
        env_test.render_df.to_csv(test_render_handle)
    return json(
        {
            "status": "success",
            "model_name": model_handle,
            "test_render": test_render_handle,
        }
    )


@main_app.get("/stop")
def stop_handler(request: Request):
    """
    Stop the training for a specific job.

    Args:
        request (Request): The request object.

    Returns:
        dict: A dictionary indicating the status of the operation.
    """
    jobname = request.args.get("jobname", "default")
    logger.info("Stopping training for job: %s", jobname)
    request.app.shared_ctx.hallpass.acquire()
    logger.info("Acquired hallpass")
    try:
        workers = db.get_workers_by_name(jobname)
        request.app.m.terminate_worker(workers)
    except sanic.SanicException as e:
        logger.error("Error stopping workers")
        logger.error(e)
        return json({"status": "error"})
    finally:
        request.app.shared_ctx.hallpass.release()
        logger.info("Released hallpass")
    return json({"status": "success"})


@reactpy.component
def button():
    """
    Renders a button element.

    Returns:
        reactpy.html.button: The rendered button element.
    """
    logger.info("Rendering button")
    return reactpy.html.button({"on_click": start_handler}, "Click me!")


@reactpy.component
def react_app():
    """
    This function returns a React app.
    """
    return reactpy.html.div({}, button())


configure(main_app, react_app)

if __name__ == "__main__":
    main_app.run(host="0.0.0.0", port=8000, debug=True, workers=16)
