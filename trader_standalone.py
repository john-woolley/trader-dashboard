import argparse
from src.trader import Trader
import logging
import os
import numpy as np
from functools import partial
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--table_name", type=str, required=True)
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--jobname", type=str, default="default")
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    logger.info("Starting training")
    table_name = args.table_name
    ncpu = args.ncpu
    jobname = args.jobname
    render_mode = args.render_mode
    timesteps = args.timesteps
    cv_periods = 5
    logger.info(
        "Starting training with cv_periods=%s, timesteps=%s, ncpu=%s, jobname=%s",
        cv_periods,
        timesteps,
        ncpu,
        jobname,
    )
    for i in range(0, cv_periods - 1):
        logger.info("Training on fold %i of %i", i, cv_periods)
        env_fn = partial(T rader, table_name, i, test=True, render_mode=render_mode)
        logger.info("Acquired hallpass")
        try:
            env_fns = [env_fn for _ in range(ncpu)]
            env = SubprocVecEnv(env_fns)  # type: ignore
            env = VecMonitor(env, log_dir)  # type: ignore
        except MemoryError as e:
            logger.error("Out of memory error creating environment")
            logger.error(e)
        policy_kwargs = {
            "net_arch": {
                "pi": [4096, 2048, 1024],
                "vf": [4096, 2048, 1024],
                "qf": [4096, 2048, 1024],
            }
        }
        model_train = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            batch_size=512,
            use_sde=True,
        )
        model_train.learn(total_timesteps=timesteps, progress_bar=True)
        model_train.save(f"model_{i}")
        env_test = Trader(table_name, i + 1, test=True, render_mode=render_mode)
        model_handle = f"model_{i}"
        model_test = SAC.load(model_handle, env=env_test)
        vec_env = model_test.get_env()
        assert isinstance(vec_env, DummyVecEnv)
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
