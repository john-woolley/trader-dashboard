# Description: Main file for the trader dashboard application.
"""
This file contains the main code for the trader dashboard application.
It sets up the Sanic server, handles requests for uploading CSV files,
starting training, and stopping training. 
It also includes functions for creating the React components used in
the application.
"""
import asyncio
import multiprocessing as mp
import os
import gc

from functools import partial
from typing import Type

import reactpy
import sanic
import numpy as np
import polars as pl
import polars.selectors as cs
import torch

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from reactpy.backend.sanic import configure
from reactpy_apexcharts import ApexChart
from sanic.log import logger
from sanic import Request, json, Sanic, redirect, html

from src import db
from src.sanic_vec_env import SanicVecEnv
from src.trader import Trader
from src.render_gfx import get_rewards_curve_figure
from src.memoryqueue import MemoryQueue, Job

Sanic.start_method = "fork"
main_app = Sanic(__name__)
app_path = os.path.dirname(__file__)
main_app.static("/static", os.path.join(app_path, "static"))
log_dir = os.path.join(app_path, "log")



@main_app.main_process_start
async def main_process_start(app):
    sanic_ver = sanic.__version__
    logger.info("Created server process running Sanic Version %s", sanic_ver)
    logger.debug("Main process started")
    app.shared_ctx.mp_ctx = mp.get_context("spawn")
    logger.debug("Created shared context")
    app.shared_ctx.hallpass = mp.Semaphore()
    app.shared_ctx.hallpass.acquire()
    app.shared_ctx.queue = MemoryQueue(app.shared_ctx.mp_ctx)
    await db.RenderData.drop()
    logger.debug("Dropped old render data table")
    await db.StdCVData.drop()
    logger.debug("Dropped old standard cv data table")
    await db.CVData.drop()
    logger.debug("Dropped old raw data table")
    await db.Workers.drop()
    logger.debug("Dropped old workers table")
    await db.Jobs.drop()
    logger.debug("Dropped old jobs table")
    await db.Jobs.create()
    logger.debug("Created new jobs table")
    await db.Workers.create()
    logger.debug("Created new workers table")
    await db.CVData.create()
    logger.debug("Created new cv data table")
    await db.StdCVData.create()
    logger.debug("Created new standard cv data table")
    await db.RenderData.create()
    logger.debug("Created new render data table")


@main_app.main_process_ready
async def main_process_ready(app: Sanic):
    loop = asyncio.get_running_loop()
    def gpu_allocator(gpu_queue: MemoryQueue):
        gpu_funs = {
            "manage_training": manage_training,
        }
        while True:
            next_job = gpu_queue.get()
            if next_job:
                current_gpu_usage = torch.cuda.memory_allocated()
                gpu_capacity = torch.cuda.get_device_properties(0).total_memory
                if current_gpu_usage + next_job.memory_usage <= gpu_capacity:
                    fn = gpu_funs[next_job.fn_name]
                    args = next_job.args
                    cv_periods = next_job.cv_periods
                    coro = fn(args, cv_periods, app)
                    asyncio.run(coro, loop)
                else:
                    gpu_queue.add(next_job)
    app.manager.manage(
        ident="GPUAllocator",
        func=gpu_allocator,
        kwargs={'gpu_queue': app.shared_ctx.gpu_queue}
    )
    global VecEnv 
    VecEnv = partial(SanicVecEnv, sanic_app=app)


@main_app.get("/start")
async def start_handler(request: Request):
    args = request.args
    logger.info("Starting training")
    table_name = args.get("table_name")
    jobname = args.get("jobname", "default")
    cv_periods = await db.CVData.get_cv_no_chunks(table_name)
    await db.Jobs.add(jobname)
    logger.info(
        "Starting training for %s with cv_periods=%s",
        table_name,
        cv_periods,
    )
    coro = manage_training(args, cv_periods, main_app)
    task = asyncio.get_event_loop().create_task(coro)
    return json({"status": "success"})


async def manage_training(args, cv_periods, app):
    jobname = args.get("jobname", "default")
    await db.Jobs.start(jobname)
    start_i = int(args.get("i", 0))
    for i in range(start_i, cv_periods - 1):
        logger.info("Starting training on fold %i of %i", i, cv_periods)
        chunk_job_train = f"{jobname}_train_{i}"
        chunk_job_validate = f"{jobname}_validate_{i}"
        await db.Jobs.add(chunk_job_train)
        await train_cv_period(args, i, app)
        await db.Jobs.add(chunk_job_validate)
        await validate_cv_period(args, i)

    await db.Jobs.complete(jobname)

@main_app.get("/get_job_status")
async def get_job_status(request: Request):
    jobname = request.args.get("jobname", "default")
    status = await db.Jobs.get(jobname)
    return json({"status": status})

@main_app.get("/get_jobs")
async def get_jobs(request: Request):
    jobs = await db.Jobs.get_all()
    res = {job[0]: job[1] for job in jobs}
    return json({"jobs": res})

async def train_cv_period(args, i, app):
    table_name = args.get("table_name")
    ncpu = int(args.get("ncpu", 64))
    render_mode = args.get("render_mode", None)
    network_depth = int(args.get("network_depth", 4))
    timesteps = int(args.get("timesteps", 1000))
    batch_size = int(args.get("batch_size", 1024))
    use_sde = bool(args.get("use_sde", 0))
    device = args.get("device", "auto")
    progress_bar = bool(args.get("progress_bar", 1))
    jobname = args.get("jobname", "default")
    network_depth = int(args.get("network_depth", 4))
    network_width = []
    for j in range(network_depth):
        depth_j = int(args.get(f"network_width_{j}", 4096))
        network_width.append(depth_j)
    model_name = args.get("model_name", "ppo")
    risk_aversion = float(args.get("risk_aversion", 0.9))
    model = PPO if model_name == "ppo" else SAC
    chunk_job_train = f"{jobname}_train_{i}"
    await db.Jobs.start(chunk_job_train)
    try:
        env_fn = partial(
            Trader,
            table_name,
            i,
            test=True,
            render_mode=render_mode,
            risk_aversion=risk_aversion,
        )
        env_fns = [env_fn for _ in range(ncpu)]
        env = SanicVecEnv(sanic_app=app, env_fns=env_fns, jobname=jobname)
    except torch.cuda.OutOfMemoryError:
        logger.error("Error creating environment")
    env = VecMonitor(env, log_dir + f"/monitor/{jobname}/{i}/{i}")
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
        tensorboard_log=log_dir + f"/tensorboard/{jobname}",
    )
    try:
        model_train.learn(total_timesteps=timesteps, progress_bar=progress_bar)
        model_train.save(f"model_{i}")
        del model_train
        gc.collect()
        torch.cuda.empty_cache()
    except KeyboardInterrupt:
        logger.error("Training interrupted")
        return redirect(f"/finished_training?jobname={jobname}?status=error")
    env.close()
    await db.Jobs.complete(chunk_job_train)


async def validate_cv_period(args, i: int):
    table_name = args.get("table_name")
    render_mode = args.get("render_mode", "human")
    render_mode = args.get("render_mode", None)
    model_name = args.get("model_name", "ppo")
    risk_aversion = float(args.get("risk_aversion", 0.9))
    jobname = args.get("jobname", "default")
    model: Type[PPO] | Type[SAC] = PPO if model_name == "ppo" else SAC
    chunk_job_validate = f"{jobname}_validate_{i}"
    await db.Jobs.start(chunk_job_validate)
    env_test = Trader(
        table_name,
        i + 1,
        test=True,
        render_mode=render_mode,
        risk_aversion=risk_aversion,
    )
    await env_test.start()
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
            obs,
            state=lstm_states,
            episode_start=episode_starts,
        )
        obs, _, done, _ = vec_env.step(action)
        if done:
            break
    render_data = env_test.get_render()
    await db.RenderData.add(table_name, render_data, i + 1)
    del model_test
    gc.collect()
    torch.cuda.empty_cache()
    await db.Jobs.complete(chunk_job_validate)


@main_app.post("/upload_csv")
async def upload_csv(request: Request):
    if not request.files:
        return json({"status": "error"})
    input_file = request.files.get("file")
    if not input_file:
        return json({"status": "error"})
    if input_file.type != "text/csv":
        return json({"status": "error"})
    output_path = request.args.get("output")
    ffill = bool(int(request.args.get("ffill", 0)))
    parse_dates = bool(int(request.args.get("parse_dates", 0)))
    file_path = os.path.join(app_path, "data", output_path)
    logger.info("Uploading %s to %s", input_file.name, file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(input_file.body.decode("utf-8"))
    logger.info("Uploaded %s to %s", input_file.name, file_path)
    coro = manage_csv_db_upload(file_path, output_path, ffill, parse_dates)
    task = asyncio.get_event_loop().create_task(coro)
    request.app.ctx.queue.put_nowait(task)
    return json({"status": "success"})


async def manage_csv_db_upload(file_path, output_path, ffill, parse_dates):
    logger.info("Uploading %s to %s", file_path, output_path)
    df = pl.read_csv(file_path, try_parse_dates=parse_dates)
    if "" in df.columns:
        df = df.drop("")
    if ffill:
        df = df.select(cs.all().forward_fill())
    db.RawData.insert(df, output_path)
    logger.info("Inserted %s into database as raw table", output_path)
    return json({"status": "success"})

@main_app.get("/get_test_render")
async def get_test_render(request: Request):
    jobname = request.args.get("jobname", "default")
    chunk = request.args.get("chunk", 0)
    test_render: pl.DataFrame = await db.RenderData.read(jobname, chunk)
    res = test_render.to_pandas().to_html()
    return html(res)


@main_app.get("/get_job_summary")
async def get_job_summary(request: Request):
    table_name = request.args.get("table_name")
    returns, dates = await get_validation_returns(table_name)
    returns_index = 100 * returns.cum_sum().exp()
    drawdown = 100 * (returns_index - returns_index.cum_max()) / returns_index.cum_max()
    chart = await get_returns_drawdown_chart(returns_index, drawdown, dates)
    sharpe_ratio = returns.mean() / returns.std()
    return json(
        {
            "sharpe_ratio": sharpe_ratio * np.sqrt(252),
            "returns_index": returns_index.to_list(),
        }
    )


async def get_returns_drawdown_chart(
    returns_index: pl.Series, drawdown: pl.Series, dates: pl.Series
):
    fig = ApexChart(
        options={
            "chart": {"id": "returns-drawdown-chart"},
            "xaxis": {"category": dates.to_list()},
        },
        series=[
            {"name": "returns", "data": returns_index.to_list()},
            {"name": "drawdown", "data": drawdown.to_list()},
        ],
        chart_type="line",
        width=800,
        height=400,
    )
    return fig


async def get_validation_returns(table_name):
    no_chunks = await db.CVData.get_cv_no_chunks(table_name)
    accumulator = []
    date_accumulator = []
    for i in range(1, no_chunks):
        render = await db.RenderData.read(table_name, i)
        render = render.with_columns(
            pl.col("market_value").log().diff().alias("returns")
        )
        returns = render["returns"]
        dates = render["Date"]
        accumulator.append(returns)
        date_accumulator.append(dates)
    returns = pl.concat(accumulator)
    dates = pl.concat(date_accumulator)
    return returns, dates


@main_app.get("/prepare_data")
async def prepare_data(request: Request):
    logger.info("Preparing data")
    table_name = request.args.get("table_name")
    no_chunks = int(request.args.get("no_chunks", 0))
    chunk_size = int(request.args.get("chunk_size", 1000))
    coro = manage_prepare_data(table_name, no_chunks, chunk_size)
    task = asyncio.get_event_loop().create_task(coro)
    return json({"status": "success"})


async def manage_prepare_data(table_name, no_chunks, chunk_size):
    logger.info("Preparing data")
    if not no_chunks:
        await db.RawData.chunk(table_name, chunk_size=chunk_size)
        no_chunks = await db.CVData.get_cv_no_chunks(table_name)
    else:
        await db.RawData.chunk(table_name, no_chunks=no_chunks)
    coros = [db.CVData.standardize(table_name, i) for i in range(no_chunks)]
    asyncio.gather(*coros)
    return json({"status": "success"})


@main_app.get("/stop")
def stop_handler(request: Request):
    jobname = request.args.get("jobname", "default")
    logger.info("Stopping training for job: %s", jobname)
    request.app.shared_ctx.hallpass.acquire()
    logger.info("Acquired hallpass")
    try:
        workers = db.Workers.get_workers_by_name(jobname)
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
    logger.info("Rendering button")
    return reactpy.html.button({"on_click": start_handler}, "Click me!")


@reactpy.component
def react_app():
    """
    This function returns a React app.
    """
    return reactpy.html.div(
        get_rewards_curve_figure(log_dir + "/monitor/TEST/1"), button()
    )


configure(main_app, react_app)

if __name__ == "__main__":
    main_app.run(host="0.0.0.0", port=8004, debug=True, workers=16)
