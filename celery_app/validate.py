import logging
from memcache_lock import memcache_lock
from trader import Trader
import click
import numpy as np
import torch
import gc
import zipfile
import db
from stable_baselines3 import PPO, SAC
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--table_name", default="trader", help="Name of the table to use")
@click.option("--jobname", default="test", help="Job name")
@click.option("--i", default=0, help="Fold")
@click.option("--render_mode", default="none", help="Render mode")
@click.option("--risk_aversion", default=0.9, help="Risk aversion")
@click.option("--model_name", default="sac", help="Model name")
def validate(
    table_name: str,
    jobname: str,
    i: int,
    render_mode: str = "none",
    risk_aversion: float = 0.9,
    model_name: str = "ppo",
) -> None:
    model: type[PPO] | type[SAC] = PPO if model_name == "ppo" else SAC
    env_test = Trader(
        table_name,
        jobname,
        i + 1,
        test=True,
        render_mode=render_mode,
        risk_aversion=risk_aversion,
    )
    model_handle = f"model_{jobname}_{i}/best_model"
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

if __name__ == "__main__":
    validate()