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

CONN = "postgresql+psycopg2://trader_dashboard@0.0.0.0:5432/trader_dashboard"
main_app = Sanic(__name__)
main_app.config["RESPONSE_TIMEOUT"] = 3600
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
    with open(input_path, "r", encoding="utf-8") as f:
        input_file = f.read()
    if not output_path:
        output_path = input_path.split("/")[-1].split(".")[0]
    file_path = os.path.join(app_path, "data", output_path)
    logger.info("Uploading %s to %s", input_path, file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(input_file)
    logger.info("Uploaded %s to %s", input_path, file_path)
    df = pd.read_csv(file_path, parse_dates=parse_dates, index_col=index_col)
    db.insert_raw_table(df, output_path)
    logger.info("Inserted %s into database as raw table", output_path)
    return json({"status": "success"})


@main_app.get("/get_jobs")
def get_jobs(request: Request):
    """
    Get all jobs from the database.

    Args:
        request (Request): The request object.

    Returns:
        dict: A dictionary containing the jobs.
    """
    jobs = db.get_jobs()
    return json({"jobs": jobs})


@main_app.get("/get_workers")
def get_workers(request: Request):
    """
    Get all workers from the database.

    Args:
        request (Request): The request object.

    Returns:
        dict: A dictionary containing the workers.
    """
    workers = db.get_workers()
    return json({"workers": workers})


@main_app.get("/get_test_render")
def get_test_render(request: Request):
    """
    Get the test render file for a specific job.

    Args:
        request (Request): The request object.

    Returns:
        dict: A dictionary containing the test render file.
    """
    jobname = request.args.get("jobname", "default")
    test_render = db.get_test_render(jobname)
    return json({"test_render": test_render})


@main_app.get("/get_model")
def get_model(request: Request):
    """
    Get the model file for a specific job.

    Args:
        request (Request): The request object.

    Returns:
        dict: A dictionary containing the model file.
    """
    jobname = request.args.get("jobname", "default")
    model = db.get_model(jobname)
    return json({"model": model})


@main_app.get("/get_log")
def get_log(request: Request):
    """
    Get the log file for a specific job.

    Args:
        request (Request): The request object.

    Returns:
        dict: A dictionary containing the log file.
    """
    jobname = request.args.get("jobname", "default")
    log = db.get_log(jobname)
    return json({"log": log})


@main_app.get("/prepare_data")
def prepare_data(request: Request):
    """
    Prepare the data for training.

    Args:
        request (Request): The request object.

    Returns:
        dict: A dictionary indicating the status of the operation.
    """
    logger.info("Preparing data")
    table_name = request.args.get("table_name")
    no_chunks = int(request.args.get("no_chunks", 0))
    chunk_size = int(request.args.get("chunk_size", 1000))
    if not no_chunks:
        db.create_chunked_table(table_name, chunk_size=chunk_size)
    else:
        db.create_chunked_table(table_name, no_chunks=no_chunks)
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
    args = request.args
    logger.info("Starting training")
    table_name = args.get("table_name")
    start_date = args.get("train_start_date")
    ncpu = int(args.get("ncpu", 1))
    jobname = args.get("jobname", "default")
    render_mode = args.get("render_mode", None)
    network_depth = int(args.get("network_depth", 4))
    batch_size = int(args.get("batch_size", 1024))
    progress_bar = bool(int(args.get("progress_bar", 1)))
    use_sde = bool(int(args.get("use_sde", 0)))
    device = args.get("device", "auto")
    train_freq = int(args.get("train_freq", 1))
    network_width = []
    for i in range(network_depth):
        network_width.append(int(request.args.get(f"network_width_{i}", 4096)))
    network = {
        "pi": network_width,
        "vf": network_width,
        "qf": network_width,
    }
    logger.info("Network architecture: %s", network)
    assert network is not None
    assert isinstance(network, dict)
    db.add_job(jobname)
    timesteps = int(request.args.get("timesteps", 1000))
    cv_periods = 5
    logger.info(
        "Starting training with cv_periods=%s and train_start_date=%s",
        cv_periods,
        start_date,
    )
    for i in range(0, cv_periods - 1):
        logger.info("Training on fold %i of %i", i, cv_periods)
        env_fn = partial(Trader, table_name, i, test=True, render_mode=render_mode)
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
            "net_arch": network,
        }
        model_train = model(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            batch_size=batch_size,
            use_sde=use_sde,
            device=device,
            tensorboard_log=log_dir,
            train_freq=train_freq,
        )
        model_train.learn(total_timesteps=timesteps, progress_bar=progress_bar)
        model_train.save(f"model_{i}")
        env.close()
        env_test = Trader(table_name, i + 1, test=True, render_mode=render_mode)
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
        while True:
            action, lstm_states = model_test.predict(
                #  We ignore the type error here because the type hint for obs
                #  is not correct
                obs,  # type: ignore
                state=lstm_states,
                episode_start=episode_starts,
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
