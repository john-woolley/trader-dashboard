import click
import logging
from stable_baselines3 import PPO, SAC
from vec_env import CeleryVecEnv
from callbacks import UpdatePctCallback
from memcache_lock import memcache_lock
from trader import Trader
from typing import Type
import zipfile
import torch
import gc
import time
from functools import partial

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
@click.option("--ncpu", default=4, help="Number of CPUs to use")
@click.option("--render_mode", default="none", help="Render mode")
@click.option("--network_depth", default=2, help="Network depth")
@click.option("--network_width", default=256, help="Network width")
@click.option("--timesteps", default=100000, help="Number of timesteps")
@click.option("--batch_size", default=64, help="Batch size")
@click.option("--use_sde", default=False, help="Use SDE")
@click.option("--device", default="auto", help="Device")
@click.option("--jobname", default="test", help="Job name")
@click.option("--model_name", default="sac", help="Model name")
@click.option("--risk_aversion", default=0.9, help="Risk aversion")
@click.option("--cv_periods", default=5, help="Cross validation periods")
@click.option("--i", default=0, help="Fold")
def train(
    table_name: str,
    jobname: str,
    i: int,
    cv_periods: int,
    timesteps: int = 1,
    ncpu: int = 1,
    render_mode: str = "none",
    network_depth: int = 2,
    network_width: int = 256,
    batch_size: int = 64,
    use_sde: bool = False,
    device: str = "auto",
    model_name: str = "sac",
    risk_aversion: float = 0.9,
):
    model: Type[PPO] | Type[SAC] = PPO if model_name == "ppo" else SAC
    chunk_job_train = f"{jobname}.train.{i}"
    logger.info("Starting training on fold %i of %i of job %s", i, cv_periods, jobname)
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
                env = CeleryVecEnv(chunk_job_train, env_fns)
    except torch.cuda.OutOfMemoryError:
        logger.error("Error creating environment")

    network = {
        "pi": [network_width] * network_depth,
        "vf": [network_width] * network_depth,
        "qf": [network_width] * network_depth,
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
    callback = UpdatePctCallback(
        env,
        total_timesteps=timesteps,
        jobname=chunk_job_train,
        best_model_save_path=f"model_{jobname}_{i}",
        log_path=f"log/eval/{jobname}",
        eval_freq=20,
        n_eval_episodes=1
        )
    model_train.learn(total_timesteps=timesteps, callback=callback)
    ensure_zip(f"model_{jobname}_{i}/best_model")
    with memcache_lock() as acquired:
        if acquired:
            del model_train
            gc.collect()
            torch.cuda.empty_cache()
    logger.info("Training on fold %i of %i of job %s complete", i, cv_periods, jobname)


if __name__ == "__main__":
    train()
