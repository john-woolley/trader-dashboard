import numpy as np
import torch
import gc
import time

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

logger = logging.getLogger(__name__)


LOCK_EXPIRE = 60 * 10  # Lock expires in 10 minutes

@contextmanager
def memcache_lock():
    timeout_at = time.monotonic() + LOCK_EXPIRE - 3
    # cache.add fails if the key already exists
    status = cache.set("train_cpu_dump", LOCK_EXPIRE)
    try:
        yield status
    finally:
        # memcache delete is very slow, but we have to use it to take
        # advantage of using add() for atomic locking
        if time.monotonic() < timeout_at and status:
            # don't release the lock if we exceeded the timeout
            # to lessen the chance of releasing an expired lock
            # owned by someone else
            # also don't release the lock if we didn't acquire it
            cache.delete("train_cpu_dump")



@celery_app.task
def manage_training(args, cv_periods):
    args = {k: v[0] for k, v in args.items() if v is not None}
    jobname = args.get("jobname", "default")
    db.Jobs.start(jobname)
    start_i = int(args.get("i", 0))
    for i in range(start_i, cv_periods - 1):
        chunk_job_train = f"{jobname}_train_{i}"
        chunk_job_validate = f"{jobname}_validate_{i}"
        db.Jobs.add(chunk_job_train)
        db.Jobs.add(chunk_job_validate)
    for i in range(start_i, cv_periods - 1):
        logger.info("Starting training on fold %i of %i of job %s",
                     i, cv_periods, jobname)
        train_cv_period(args, i)
        validate_cv_period(args, i)

    db.Jobs.complete(jobname)


def train_cv_period(args, i):
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
    db.Jobs.start(chunk_job_train)
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


def validate_cv_period(args, i: int):
    table_name = args.get("table_name")
    render_mode = args.get("render_mode", "human")
    render_mode = args.get("render_mode", None)
    model_name = args.get("model_name", "ppo")
    risk_aversion = float(args.get("risk_aversion", 0.9))
    jobname = args.get("jobname", "default")
    model: Type[PPO] | Type[SAC] = PPO if model_name == "ppo" else SAC
    chunk_job_validate = f"{jobname}_validate_{i}"
    db.Jobs.start(chunk_job_validate)
    env_test = Trader(
        table_name,
        i + 1,
        test=True,
        render_mode=render_mode,
        risk_aversion=risk_aversion,
    )
    model_handle = f"model_{jobname}_{i}"
    with memcache_lock() as acquired:
        if acquired:
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
    db.RenderData.add(table_name, render_data, i + 1)
    with memcache_lock() as acquired:
        if acquired:
            del model_test
            gc.collect()
            torch.cuda.empty_cache()
            db.Jobs.complete(chunk_job_validate)
