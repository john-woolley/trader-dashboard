import time
import logging
from celery import group, chain
import subprocess
from celery_app import celery_app
import db

logger = logging.getLogger(__name__)


@celery_app.task
def complete_job_async(result, jobname):
    db.Jobs.complete(jobname)


@celery_app.task
def add_render_data_async(*args):
    table_name = args[-4]
    render_data = args[-3]
    i = args[-2]
    jobname = args[-1]
    db.RenderData.add(table_name, render_data, i, jobname)


@celery_app.task
def standardize_async(*args):
    table_name = args[-3]
    jobname = args[-2]
    i = args[-1]
    db.CVData.standardize(table_name, jobname, i)


@celery_app.task
def sleep(nil_res, seconds):
    time.sleep(seconds)
    return seconds


@celery_app.task
def manage_training(args):
    args = {k: v[0] for k, v in args.items() if v is not None}
    
    jobname = args.get("jobname")
    table_name = args.get("table_name")
    db.Jobs.start(jobname)
    start_i = int(args.get("i", 0))
    cv_periods = db.CVData.get_cv_no_chunks(table_name, jobname) - 1

    for i in range(start_i, cv_periods):
        chunk_job_train = f"{jobname}_train_{i}"
        chunk_job_validate = f"{jobname}_validate_{i}"
        db.Jobs.add(chunk_job_train, parent=jobname)
        db.Jobs.add(chunk_job_validate, parent=jobname)
    
    for i in range(start_i, cv_periods):
        train_cv_period(args, i, cv_periods)
        validate_cv_period(args, i, cv_periods)
    
    db.Jobs.complete(jobname)


@celery_app.task
def manage_training_async(args):
    args = {k: v[0] for k, v in args.items() if v is not None}
    max_concurrency = int(args.get("max_concurrency", 4))
    jobname = args.get("jobname", "default")
    db.Jobs.start(jobname)
    start_i = int(args.get("i", 0))
    table_name = args.get("table_name")
    cv_periods = db.CVData.get_cv_no_chunks(table_name, jobname) - 1

    group_tasks = [sleep.s(0, 0)]
    i = start_i
    while i < cv_periods:

        chain_tasks = []
        j = 0
        while j < max_concurrency and i < cv_periods:
        
            chunk_job_train = f"{jobname}.train.{i}"
            chunk_job_validate = f"{jobname}.validate.{i}"
            db.Jobs.add(chunk_job_train, parent=jobname)
            logger.info("Adding job %s", chunk_job_train)
            db.Jobs.add(chunk_job_validate, parent=jobname)
            logger.info("Adding job %s", chunk_job_validate)
            chain_tasks.append(
                chain(
                    train_cv_period_chained.s(args, i, cv_periods),
                    validate_cv_period_chained.s(args, i, cv_periods),
                )
            )
            logger.info("Adding tasks to chain")
            i += 1
            j += 1

        group_tasks.append(group(chain_tasks))
        logger.info("Adding chain to group")

    job = chain(*group_tasks) | complete_job_async.s(jobname)
    logger.info("Job %s started", jobname)
    job.apply_async()


@celery_app.task()
def train_cv_period(args, i: int, cv_periods: int):
    table_name = args.get("table_name")
    ncpu = int(args.get("ncpu", 64))
    render_mode = args.get("render_mode", None)
    network_depth = int(args.get("network_depth", 4))
    timesteps = int(args.get("timesteps", 1000))
    batch_size = int(args.get("batch_size", 1024))
    use_sde = bool(args.get("use_sde", 0))
    device = args.get("device", "auto")
    jobname = args.get("jobname", "default")
    network_depth = int(args.get("network_depth", 4))
    network_width = []
    for j in range(network_depth):
        depth_j = int(args.get(f"network_width_{j}", 4096))
        network_width.append(depth_j)
    model_name = args.get("model_name", "ppo")
    risk_aversion = float(args.get("risk_aversion", 0.9))
    chunk_job_train = f"{jobname}.train.{i}"
    status = db.Jobs.get(chunk_job_train)
    if status != "pending":
        return
    logger.info("Starting training on fold %i of %i of job %s", i, cv_periods, jobname)
    db.Jobs.start(chunk_job_train)
    subprocess.call(
        [
            "python",
            "trader_dashboard/celery_app/train.py",
            "--table_name",
            table_name,
            "--ncpu",
            str(ncpu),
            "--render_mode",
            render_mode,
            "--network_depth",
            str(network_depth),
            "--timesteps",
            str(timesteps),
            "--batch_size",
            str(batch_size),
            "--use_sde",
            str(use_sde),
            "--device",
            device,
            "--jobname",
            jobname,
            "--model_name",
            model_name,
            "--risk_aversion",
            str(risk_aversion),
            "--i",
            str(i),
        ]
    )
    db.Jobs.complete(chunk_job_train)


@celery_app.task()
def validate_cv_period(args, i: int, cv_periods: int):
    table_name = args.get("table_name")
    render_mode = args.get("render_mode", "human")
    render_mode = args.get("render_mode", None)
    model_name = args.get("model_name", "ppo")
    risk_aversion = float(args.get("risk_aversion", 0.9))
    jobname = args.get("jobname", "default")
    chunk_job_validate = f"{jobname}.validate.{i}"
    status = db.Jobs.get(chunk_job_validate)
    if status != "pending":
        return
    logger.info(
        "Starting validation on fold %i of %i of job %s", i, cv_periods, jobname
    )
    db.Jobs.start(chunk_job_validate)
    subprocess.call(
        [
            "python",
            "trader_dashboard/celery_app/validate.py",
            "--table_name",
            table_name,
            "--jobname",
            jobname,
            "--i",
            str(i),
            "--render_mode",
            render_mode,
            "--risk_aversion",
            str(risk_aversion),
            "--model_name",
            model_name,
        ]
    )
    db.Jobs.complete(chunk_job_validate)
    logger.info("Validation on fold %i of job %s complete", i, jobname)


@celery_app.task
def validate_cv_period_chained(*args):
    cv_periods = args[-1]
    i = args[-2]
    args = args[-3]
    return validate_cv_period(args, i, cv_periods)


@celery_app.task
def train_cv_period_chained(*args):
    cv_periods = args[-1]
    i = args[-2]
    args = args[-3]
    return train_cv_period(args, i, cv_periods)


@celery_app.task
def manage_prepare_data(args):
    args = {k: v[0] for k, v in args.items() if v is not None}
    table_name = args.get("table_name")
    jobname = args.get("jobname")
    no_chunks = int(args.get("no_chunks", 0))
    chunk_size = int(args.get("chunk_size", 0))
    logger.info("Preparing data")
    if not no_chunks:
        db.RawData.chunk(table_name, jobname, chunk_size=chunk_size)
        no_chunks = db.CVData.get_cv_no_chunks(table_name, jobname)
    else:
        db.RawData.chunk(table_name, jobname, no_chunks=no_chunks)
    for i in range(no_chunks):
        db.CVData.standardize(table_name, jobname, i)


@celery_app.task
def manage_prepare_data_async(args):
    args = {k: v[0] for k, v in args.items() if v is not None}
    table_name = args.get("table_name")
    jobname = args.get("jobname")
    no_chunks = int(args.get("no_chunks", 0))
    chunk_size = int(args.get("chunk_size", 0))
    max_concurrency = int(args.get("max_concurrency", 4))
    logger.info("Preparing data")

    if not no_chunks:
        db.RawData.chunk(table_name, jobname, chunk_size=chunk_size)
        no_chunks = db.CVData.get_cv_no_chunks(table_name, jobname)
    else:
        db.RawData.chunk(table_name, jobname, no_chunks=no_chunks)

    i = 0
    chain_tasks = [sleep.s(0, 0)]
    while i < no_chunks:

        j = 0
        group_tasks = []
        while j < max_concurrency and i < no_chunks:
            group_tasks.append(standardize_async.s(table_name, jobname, i))
            i += 1
            j += 1

        chain_tasks.append(group(*group_tasks))

    job = chain(*chain_tasks)
    job.apply_async()
