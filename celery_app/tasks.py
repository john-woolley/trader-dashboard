import asyncio
import numpy as np
import torch
import gc
import time
import zipfile

from contextlib import contextmanager
from functools import partial
from typing import Type
import logging

from celery import group, chain, chord
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv

from celery_app import celery_app
import db
from trader import Trader
from cache import cache
import logging

logger = logging.getLogger(__name__)


LOCK_EXPIRE = 60 * 10  # Lock expires in 10 minutes


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
def sleep(nil_res, seconds):
    time.sleep(seconds)
    return seconds


@contextmanager
def memcache_lock():
    timeout_at = time.monotonic() + LOCK_EXPIRE - 3
    status = cache.set("gpu_cpu_dump", LOCK_EXPIRE)
    try:
        yield status
    finally:
        if time.monotonic() < timeout_at and status:
            cache.delete("gpu_cpu_dump")

@celery_app.task
def manage_training(args):
    args = {k: v[0] for k, v in args.items() if v is not None}
    jobname = args.get("jobname")
    table_name = args.get("table_name")
    db.Jobs.start(jobname)
    start_i = int(args.get("i", 0))
    cv_periods =  db.CVData.get_cv_no_chunks(table_name, jobname)

    for i in range(start_i, cv_periods - 1):
        chunk_job_train = f"{jobname}_train_{i}"
        chunk_job_validate = f"{jobname}_validate_{i}"
        db.Jobs.add(chunk_job_train, parent=jobname)
        db.Jobs.add(chunk_job_validate, parent=jobname)
    for i in range(start_i, cv_periods - 1):
        train_cv_period(args, i, cv_periods - 1)
        validate_cv_period(args, i, cv_periods - 1)
    db.Jobs.complete(jobname)
 

@celery_app.task
def manage_training_async(args):
    args = {k: v[0] for k, v in args.items() if v is not None}
    max_concurrency = int(args.get("max_concurrency", 4))
    jobname = args.get("jobname", "default")
    db.Jobs.start(jobname)
    start_i = int(args.get("i", 0))
    table_name = args.get("table_name")
    cv_periods =  db.CVData.get_cv_no_chunks(table_name, jobname)
    
    group_tasks = []
    i = start_i
    while i < cv_periods - 1:
        chain_tasks = []
        for _ in range(max_concurrency):
            chunk_job_train = f"{jobname}_train_{i}"
            chunk_job_validate = f"{jobname}_validate_{i}"
            if i >= cv_periods - 1:
                break
            db.Jobs.add(chunk_job_train, parent=jobname)
            logger.info("Adding job %s", chunk_job_train)
            db.Jobs.add(chunk_job_validate, parent=jobname)
            logger.info("Adding job %s", chunk_job_validate)
            chain_tasks.append(
                chain(
                    train_cv_period_chained.s(args, i, cv_periods - 1),
                    validate_cv_period_chained.s(args, i, cv_periods - 1)
                )
            )
            logger.info("Adding task to job chain %s", chain_tasks[-1])
            i += 1
        group_tasks.append(group(chain_tasks))
        logger.info("Adding job group %s", group_tasks[-1])
    job = chain(*group_tasks) | complete_job_async.s(jobname)
    job.apply_async()
    logger.info("Job %s started", job)


@celery_app.task(bind=True)
def train_cv_period(self, args, i: int, cv_periods: int):
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
    model = PPO if model_name == "ppo" else SAC
    chunk_job_train = f"{jobname}_train_{i}"
    logger.info("Starting training on fold %i of %i of job %s", i, cv_periods, jobname)
    db.Jobs.start(chunk_job_train)
    try:
        env_fn = partial(
            Trader,
            table_name,
            jobname,
            i,
            test=True,
            render_mode=render_mode,
            risk_aversion=risk_aversion,
        )
        env_fns = [env_fn for _ in range(ncpu)]
        with memcache_lock() as acquired:
            if acquired:
                env = SubprocVecEnv(env_fns)
    except torch.cuda.OutOfMemoryError:
        logger.error("Error creating environment")
    env = VecMonitor(env, f"log/monitor/{jobname}/{i}/{i}")
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
        tensorboard_log=f"log/tensorboard/{jobname}",
    )
    model_train.learn(total_timesteps=timesteps)
    with memcache_lock() as acquired:
        if acquired:
            model_train.save(f"model_{jobname}_{i}")
            del model_train
            gc.collect()
            torch.cuda.empty_cache()
    db.Jobs.complete(chunk_job_train)
    logger.info("Training on fold %i of %i of job %s complete", i, cv_periods, jobname)
    asyncio.sleep(0.1)

def ensure_zip(file_handle):
    try:
        with zipfile.ZipFile(file_handle + '.zip', 'r') as zip_file:
            zip_file.testzip()
        return True

    except zipfile.BadZipFile:
        logger.error("The file handle is not a valid CRC zip file.")
        return False
    
    except FileNotFoundError:
        logger.error("The file handle is not a valid file.")
        return False


@celery_app.task(bind=True)
def validate_cv_period(self, args, i: int, cv_periods:int):
    table_name = args.get("table_name")
    render_mode = args.get("render_mode", "human")
    render_mode = args.get("render_mode", None)
    model_name = args.get("model_name", "ppo")
    risk_aversion = float(args.get("risk_aversion", 0.9))
    jobname = args.get("jobname", "default")
    model: Type[PPO] | Type[SAC] = PPO if model_name == "ppo" else SAC
    chunk_job_validate = f"{jobname}_validate_{i}"
    logger.info("Starting validation on fold %i of %i of job %s", i, cv_periods, jobname)
    db.Jobs.start(chunk_job_validate)
    env_test = Trader(
        table_name,
        jobname,
        i + 1,
        test=True,
        render_mode=render_mode,
        risk_aversion=risk_aversion,
    )
    model_handle = f"model_{jobname}_{i}"
    if not ensure_zip(model_handle):
        logger.error("Model file not found")
        db.Jobs.complete(chunk_job_validate)
        return
    with memcache_lock() as acquired:
        if acquired:
            if not ensure_zip(model_handle):
                logger.error("Model file not found")
                db.Jobs.complete(chunk_job_validate)
                return
            model_test = model.load(model_handle, env=env_test)
    vec_env = model_test.get_env()
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
    db.RenderData.add(table_name, render_data, i + 1, jobname)
    with memcache_lock() as acquired:
        if acquired:
            del model_test
            gc.collect()
            torch.cuda.empty_cache()
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
