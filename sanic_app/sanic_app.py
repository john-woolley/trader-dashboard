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
import time
import subprocess

import reactpy
import sanic
import numpy as np
import polars as pl
import polars.selectors as cs
import torch
import nvidia_smi

from celery import Celery
from reactpy.backend.sanic import configure
from reactpy_apexcharts import ApexChart
from sanic.log import logger
from sanic import Request, json, Sanic, html

import db
from src.render_gfx import get_rewards_curve_figure
from src.memoryqueue import MemoryQueue, Job

celery_app = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
)

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
    app.ctx.queue = asyncio.Queue()
    app.shared_ctx.gpu_queue = MemoryQueue(app.shared_ctx.mp_ctx)
    db.Workers.drop()
    logger.debug("Dropped old workers table")
    db.Workers.create()
    logger.debug("Created new workers table")


def get_gpu_usage():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return info.used


def exponential_backoff(retry_count, base_delay=60, max_delay=3600):
    delay = min(base_delay * (2**retry_count), max_delay)
    return delay


def gpu_allocator(gpu_queue: MemoryQueue, hallpass: mp.Semaphore):
    retry_count = 0
    while True:
        next_job = gpu_queue.get()
        if next_job:
            try:
                torch.cuda.empty_cache()
                gpu_capacity = torch.cuda.get_device_properties(0).total_memory
                gpu_capacity *= 0.9
                estimated_usage = gpu_queue.estimated_memory_usage.value
                current_gpu_usage = max(get_gpu_usage(), estimated_usage)
                if current_gpu_usage + next_job.memory_usage <= gpu_capacity:
                    hallpass.acquire()
                    try:
                        fn = next_job.fn_name
                        args = next_job.args
                        celery_app.send_task(fn, args=(args,))
                        retry_count = 0
                    except torch.cuda.OutOfMemoryError:
                        logger.error("Error processing job")
                        retry_count += 1
                        gpu_queue.add(next_job)
                    finally:
                        hallpass.release()
                else:
                    logger.info(
                        (
                            "Insufficient GPU memory,"
                            f"waiting for next job. Retry count: {retry_count}"
                        )
                    )
                    gpu_queue.add(next_job)
                    time.sleep(exponential_backoff(retry_count))
                    retry_count += 1
            except Exception as e:
                logger.error(f"Error processing job: {str(e)}")
                retry_count += 1


@main_app.main_process_ready
async def main_process_ready(app: Sanic):
    app.manager.manage(
        ident="GPUAllocator",
        func=gpu_allocator,
        kwargs={
            "gpu_queue": app.shared_ctx.gpu_queue,
            "hallpass": app.shared_ctx.hallpass,
        },
    )


def estimate_gpu_usage(
    table_name, jobname, network_depth, network_width, buffer_len
) -> int:
    data = db.StdCVData.read(table_name, jobname, 0)

    no_symbols = len(data.select("ticker").unique().collect().to_series())
    no_features = 44 * no_symbols + 39
    no_actions = no_symbols + 2
    no_obs = buffer_len * no_features
    no_parameters = (no_obs + no_actions) + network_width**network_depth
    sizeof_float = 64

    return no_parameters * sizeof_float


@main_app.get("/start")
async def start_handler(request: Request):
    args = request.args

    table_name = args.get("table_name")
    jobname = args.get("jobname", "default")
    _async = bool(int(args.get("async", 0)))
    network_depth = int(args.get("network_depth", 4))
    network_width = max(
        [int(args.get(f"network_width_{i}", 4096)) for i in range(network_depth)]
    )

    cv_periods = db.CVData.get_cv_no_chunks(table_name, jobname)

    gpu_usage = estimate_gpu_usage(
        table_name, jobname, network_depth, network_width, 10
    )

    # Check if the training data exists.
    try:
        db.StdCVData.read(table_name, jobname, 0)
    except Exception as e:
        logger.error("Error reading data: %s", str(e))
        return json({"status": "error"})

    logger.info(
        "Starting training for %s with cv_periods=%s",
        table_name,
        cv_periods,
    )

    fn_name = "tasks.manage_training" if not _async else "tasks.manage_training_async"

    gpu_job = Job(
        job_id=jobname,
        memory_usage=0,
        fn_name=fn_name,
        args=args,
        cv_periods=cv_periods,
    )

    request.app.shared_ctx.gpu_queue.add(gpu_job)

    return json({"status": "success"})


@main_app.get("/get_job_status")
async def get_job_status(request: Request):
    jobname = request.args.get("jobname", "default")
    status = db.Jobs.get(jobname)
    return json({"status": status})


@main_app.get("/get_jobs")
async def get_jobs(request: Request):
    jobs = db.Jobs.get_all()

    def build_tree(job):
        children = [build_tree(child) for child in jobs if child[-1] == job[0]]
        res = {
            "name": job[0],
            "status": job[1],
            "pct_complete": job[2] or 0,
            "start_time": str(job[3]),
            "end_time": str(job[4]),
            "children": children
        }
        return {k: v for k, v in res.items() if v}

    res = [build_tree(job) for job in jobs if job[-1] is None]

    return json({"jobs": res})


@main_app.post("/upload_csv")
async def upload_csv(request: Request):
    if not request.files:
        return json({"status": "error"})

    input_file = request.files.get("file")
    if not input_file or input_file.type != "text/csv":
        return json({"status": "error"})

    output_path = request.args.get("output")
    ffill = bool(int(request.args.get("ffill", 0)))

    file_path = os.path.join(app_path, "data", output_path)

    logger.info("Uploading %s to %s", input_file.name, file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(input_file.body.decode("utf-8"))
    logger.info("Uploaded %s to %s", input_file.name, file_path)

    coro = manage_csv_db_upload(file_path, output_path, ffill)
    loop = asyncio.get_event_loop()
    task = asyncio.run_coroutine_threadsafe(coro, loop=loop)
    request.app.ctx.queue.put_nowait(task)

    return json({"status": "success"})


async def manage_csv_db_upload(file_path, output_path, ffill):
    logger.info("Uploading %s to %s", file_path, output_path)
    df = pl.read_csv(file_path, try_parse_dates=True, infer_schema_length=60000)

    def convert_date_format(date_str, output_format='%Y-%m-%d'):
        # Split the input date string using '/'
        month, day, year = map(int, date_str.split('/'))
        
        # Rearrange the components based on the output format
        formatted_date = (
            output_format.replace('%Y', str(year))
            .replace('%m', f'{month:02d}')
            .replace('%d', f'{day:02d}')
            )

        return formatted_date
    
    df = df.with_columns(pl.col("date").map_elements(convert_date_format).alias("date"))
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
    table_name = request.args.get("table_name")
    chunk = request.args.get("chunk", 0)
    test_render: pl.LazyFrame = db.RenderData.read(table_name, jobname, chunk)
    res = test_render.collect().to_pandas().to_html()
    return html(res)


@main_app.get("/get_job_summary")
async def get_job_summary(request: Request):
    table_name = request.args.get("table_name")
    jobname = request.args.get("jobname", "default")

    returns, dates = await get_validation_returns(table_name, jobname)
    sharpe_ratio = returns.mean() / returns.std()
    returns_index = 100 * returns.cum_sum().exp()
    drawdown = 100 * (returns_index - returns_index.cum_max()) / returns_index.cum_max()

    chart = await get_returns_drawdown_chart(returns_index, drawdown, dates)

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


async def get_validation_returns(table_name, jobname):
    no_chunks = db.CVData.get_cv_no_chunks(table_name, jobname)
    accumulator = []
    date_accumulator = []

    for i in range(1, no_chunks - 1):
        render = db.RenderData.read(table_name, jobname, i)
        render = render.with_columns(
            pl.col("market_value").log().diff().alias("returns")
        ).collect()
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

    jobname = request.args.get("jobname", "default")
    table_name = request.args.get("table_name")
    no_chunks = int(request.args.get("no_chunks", 0))
    chunk_size = int(request.args.get("chunk_size", 1000))
    max_concurrency = int(request.args.get("max_concurrency", 16))

    db.Jobs.add(jobname)

    coro = manage_prepare_data(
        table_name, no_chunks, chunk_size, jobname, max_concurrency
    )
    loop = asyncio.get_event_loop()
    task = asyncio.run_coroutine_threadsafe(coro, loop=loop)
    request.app.ctx.queue.put_nowait(task)

    return json({"status": "success"})


async def manage_prepare_data(
    table_name, no_chunks, chunk_size, jobname, max_concurrency=16
):
    logger.info("Preparing data")

    if not no_chunks:
        db.RawData.chunk(table_name, jobname, chunk_size=chunk_size)
        no_chunks = db.CVData.get_cv_no_chunks(table_name, jobname)
    else:
        db.RawData.chunk(table_name, jobname, no_chunks=no_chunks)

    i = 0
    while i < no_chunks:
        ops = []
        j = 0
        while j < max_concurrency and i < no_chunks:
            ops.append(db.CVData.standardize_async(table_name, jobname, i))
            i += 1
            j += 1

        ops.append(db.CVData.standardize_async(table_name, jobname, i))
        await asyncio.gather(*ops)


@main_app.get("/delete_job")
async def delete_job(request: Request):
    jobname = request.args.get("jobname", "default")
    db.Jobs.delete(jobname)
    logger.info("Deleted job %s", jobname)

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
    subprocess.run(["python", "db.py"])
    main_app.run(host="0.0.0.0", port=8004, debug=True, fast=True)
