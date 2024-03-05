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


def ensure_zip(file_handle, max_attempts=10, interval=1):
    max_attempts = 10
    attempts = 0

    while attempts < max_attempts:
        try:
            with zipfile.ZipFile(file_handle + ".zip", "r") as zip_file:
                zip_file.testzip()
            return True

        except zipfile.BadZipFile:
            logger.error("The file handle is not a valid CRC zip file.")
            return False

        except FileNotFoundError:
            logger.error("The file handle is not a valid file.")
            return False

        except Exception as e:
            logger.warning(f"Error checking zip file: {e}")

        attempts += 2
        time.sleep(interval)

    logger.error(f"Exceeded maximum attempts to check zip file: {file_handle}")
    return False


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
    with memcache_lock() as acquired:
        if acquired:
            if not ensure_zip(model_handle):
                logger.error("Model file not found")
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

if __name__ == "__main__":
    validate()