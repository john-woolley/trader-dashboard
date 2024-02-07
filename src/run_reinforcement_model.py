import asyncio
from polars import Unknown
from stable_baselines3 import SAC, PPO    
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from functools import partial
import sys
sys.path.append('/home/augustus/trading/skfolio/skfolio')
from src.ingestion import CVIngestionPipeline
from src.trader import Trader
from os import cpu_count
from stable_baselines3.common.callbacks import EvalCallback
from sanic.log import logger
import pandas as pd
from typing import Union, Callable
import gymnasium as gym
import numpy as np
CPU_COUNT = cpu_count()


def get_vec(train_data: pd.DataFrame, cpu_count: Union[int, None] = None):
    n_cpu = cpu_count or 1
    n_cpu = int(n_cpu)
    def my_partial() -> Callable[[], gym.Env]:
        return partial(Trader, train_data, test=True)
    env_vec = [my_partial() for _ in range(n_cpu)]
    return SubprocVecEnv(env_vec)



def run_reinforcement_model(model_name: str, input_file_handle: str, log_dir: str, model_file_handle: str, train_start_date: str, cpu_count: Union[str, None] = None):

    model = {'SAC': SAC, 'PPO': PPO}[model_name]
    ep_length = 252
    cv_periods = 5
    n_cpus = int(cpu_count or 1) or CPU_COUNT
    data = CVIngestionPipeline(input_file_handle, cv_periods, start_date=train_start_date)

    for i in range(len(data)):
        train, test = tuple(*iter(data))
        mfile = f"stock_allocator_{ep_length}d"
        env = get_vec(train, n_cpus)
        env = VecMonitor(env)
        eval_callback = EvalCallback(env, best_model_save_path=f"{log_dir}/{mfile}",
                            log_path=log_dir, eval_freq=500,
                            deterministic=True, render=True)
        policy_kwargs = dict(
            net_arch=dict(pi=[1024, 512], vf=[1024, 512], qf=[1024, 512])
        )
        model_train = model(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            batch_size=1024,
            use_sde=True,
        )
        model_train.learn(total_timesteps=5e5, callback=eval_callback)
        model_train.save(f"{model_file_handle}_{i}")
        env_test = Trader(test, test=True)
        model_test = model.load(
            f"{model_file_handle}_{i}", env=env_test
        )
        vec_env = model_test.get_env()
        obs = vec_env.reset()
        lstm_states = None
        num_envs = 1
        episode_starts = np.ones((num_envs,), dtype=bool)
        for j in range(len(test["close"]) - 1):
            action, lstm_states = model_test.predict(
                obs, state=lstm_states, episode_start=episode_starts
            )
            obs, reward, done, info = vec_env.step(action)
            if done:
                break
        env_test.render_df.to_csv(f"{model_file_handle}_{i}.csv")

if __name__ == '__main__':
    args = sys.argv[1:]
    run_reinforcement_model(*args)