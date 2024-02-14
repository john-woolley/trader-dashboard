# Description: Main file for the trader dashboard
"""
This file contains the main code for the trader dashboard application.
It sets up the Sanic server, handles requests for uploading CSV files,
starting training, and stopping training. 
It also includes functions for creating the React components used in
the application.

Train and validate are mutually recursive functions that train and validate
the model on a specific fold of the cross-validation period.

The train function trains the model on a specific fold of the cross-validation
period, while the validate function validates the model on a specific fold
of the cross-validation period.

The train function takes a request object containing the training parameters
and returns a continuation of the validation process.

The validate function takes a request object containing the validation parameters
and returns a continuation of the training process.

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
from sanic import Request, json, Sanic, redirect

from src import db
from src.sanic_vec_env import SanicVecEnv
from src.trader import Trader

CONN = "postgresql+psycopg2://trader_dashboard@0.0.0.0:5432/trader_dashboard"
main_app = Sanic(__name__)
main_app.config["RESPONSE_TIMEOUT"] = 3600
app_path = os.path.dirname(__file__)
main_app.static("/static", os.path.join(app_path, "static"))
log_dir = os.path.join(app_path, "log")


@main_app.get("/finished_training")
def finished_training(request: Request):
    """
    Handles the finished training request.

    Args:
        request (Request): The request object.

    Returns:
        dict: A dictionary containing the status of the operation.
    """
    jobname = request.args.get("jobname", "default")
    logger.info("Finished training for job: %s", jobname)
    return json({"status": "success"})


@main_app.get("/train_cv_period")
def train_cv_period(request):
    """
    Train the model on a specific fold of the cross-validation period.

    Args:
        request (Request): The request object containing the training parameters.

    Returns:
        A continuation of the training process.
    """

    args = request.args
    previous_worker = args.get("previous_worker", "null")
    if previous_worker != "null":
        request.app.m.restart(previous_worker)
    table_name = args.get("table_name")
    i = int(args.get("i"), 0)
    ncpu = int(args.get("ncpu", 64))
    render_mode = args.get("render_mode", None)
    network_depth = int(args.get("network_depth", 2))
    timesteps = int(args.get("timesteps", 1000))
    train_freq = int(args.get("train_freq", 1))
    batch_size = int(args.get("batch_size", 1024))
    use_sde = int(args.get("use_sde", 0))
    device = args.get("device", "auto")
    progress_bar = int(args.get("progress_bar", 1))
    jobname = args.get("jobname", "TEST")
    cv_periods = int(args.get("cv_periods", 5))
    logger.info("Training on fold %i of %i", i, cv_periods)
    main_app.shared_ctx.hallpass.acquire()
    env_fn = partial(Trader, table_name, i, test=True, render_mode=render_mode)
    env_fns = [env_fn for _ in range(ncpu)]
    env = SanicVecEnv(env_fns, request.app, jobname)  # type: ignore
    main_app.shared_ctx.hallpass.release()
    env = VecMonitor(env, log_dir)
    network_width = []
    for j in range(network_depth):
        network_width.append(int(request.args.get(f"network_width_{j}", 4096)))
    network = {
        "pi": network_width,
        "vf": network_width,
        "qf": network_width,
    }
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
    redirect_continue = request.app.url_for(
        "validate_cv_period",
        i=i,
        max_i=cv_periods-1,
        table_name=table_name,
        jobname=jobname,
        render_mode=render_mode,
        previous_worker=request.app.m.name,
        ncpu=ncpu,
        cv_periods=cv_periods,
        device=device,
        progress_bar=progress_bar,
        use_sde=use_sde,
        batch_size=batch_size,
        train_freq=train_freq,
        timesteps=timesteps,
        network_depth=network_depth,
        network_width=network_width
        )
    return redirect(redirect_continue)


@main_app.get("/validate_cv_period")
def validate_cv_period(request: Request):
    """
    Validate the table name and return the number of rows.

    Args:
        request (Request): The request object.

    Returns:
        A continuation of the training process.
    """
    args = request.args
    previous_worker = args.get("previous_worker", "null")
    if previous_worker:
        request.app.m.restart(previous_worker)
    table_name = request.args.get("table_name")
    render_mode = request.args.get("render_mode", "human")
    jobname = request.args.get("jobname", "default")
    max_i = int(request.args.get("max_i"), 5)
    i = int(request.args.get("i", 0))
    ncpu = int(request.args.get("ncpu", 64))
    render_mode = args.get("render_mode", None)
    network_depth = int(args.get("network_depth", 2))
    timesteps = int(args.get("timesteps", 1000))
    train_freq = int(args.get("train_freq", 1))
    batch_size = int(args.get("batch_size", 1024))
    use_sde = int(args.get("use_sde", 0))
    device = args.get("device", "auto")
    progress_bar = int(args.get("progress_bar", 1))
    jobname = args.get("jobname", "TEST")
    cv_periods = int(args.get("cv_periods", 5))
    network_width = args.get("network_width", [4096, 2048, 1024])
    logger.info("Validating on fold %i of %i", i + 1, cv_periods)
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
    test_render_handle = f"test_render_{i+1}.csv"
    env_test.render_df.to_csv(test_render_handle)
    i += 1
    redirect_continue = request.app.url_for(
        "train_cv_period",
        i=str(i),
        max_i=str(max_i),
        table_name=table_name,
        jobname=jobname,
        render_mode=render_mode,
        previous_worker=request.app.m.name,
        ncpu=str(ncpu),
        cv_periods=str(cv_periods),
        device=device,
        progress_bar=progress_bar,
        use_sde=use_sde,
        batch_size=batch_size,
        train_freq=train_freq,
        timesteps=timesteps,
        network_depth=network_depth,
        network_width=network_width
    )
    redirect_end = f"/finished_training?jobname={jobname}"
    if i < max_i:
        return redirect(redirect_continue)
    else:
        return redirect(redirect_end)


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
    jobname = args.get("jobname", "default")
    cv_periods = args.get("cv_periods", 5)
    db.add_job(jobname)
    logger.info(
        "Starting training for %s with cv_periods=%s",
        table_name,
        cv_periods,
    )
    request_str = request.url
    redirect_str = request_str.replace("start", "train_cv_period")
    redirect_str += f"&i=0&max_i={cv_periods-1}&jobname={jobname}"
    return redirect(redirect_str)


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
