import logging
import time
import datetime
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from cache import cache
import db
from typing import Union, Optional
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)

class UpdatePctCallback(EvalCallback):
    """
    Callback for updating the percentage of the training process.
    """

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            total_timesteps,
            jobname,
            callback_on_new_best: Optional[BaseCallback] = None,
            callback_after_eval: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
            ):
        super(UpdatePctCallback, self).__init__(
            eval_env,
            callback_on_new_best,
            callback_after_eval,
            n_eval_episodes,
            eval_freq,
            log_path,
            best_model_save_path,
            deterministic,
            render,
            verbose,
            warn,
        )
        self.jobname = jobname
        self.total_timesteps = total_timesteps
        self.progress = 0
        self.start_time = time.time()
        self.update_time = self.start_time

    def _on_step(self) -> bool:
        time_delta = time.time() - self.update_time
        self.update_time = time.time()
        progress = self.model.num_timesteps / self.total_timesteps
        progress_delta = progress - self.progress
        self.progress = progress
        eta_secs = time_delta / progress_delta * (1 - progress)
        eta = datetime.datetime.fromtimestamp(self.update_time + eta_secs)
        db.Jobs.update_pct_complete(self.jobname, progress, eta)
        logger.info("Job %s progress: %.2f%% ETA: %s", self.jobname, progress * 100, eta)
        return super(UpdatePctCallback, self)._on_step()


class RedisCallback(BaseCallback):
    """
    Callback for putting state of the training process into Redis.
    """

    def __init__(self, total_timesteps, jobname, verbose=0):
        super(RedisCallback, self).__init__(verbose)
        self.jobname = jobname
        cache.set(self.jobname + "_maxiter", total_timesteps)
        cache.set(self.jobname + "_progress", 0)

    def _on_step(self) -> bool:
        cache.set(self.jobname + "_progress", self.model.num_timesteps)
        return True


class ProgressBarCallback(BaseCallback):
    """
    Callback for displaying a slick ASCII progress bar.
    """

    def __init__(self, total_timesteps, jobname, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress = 0
        self.jobname = jobname

    def _on_step(self) -> bool:
        self.progress = self.model.num_timesteps
        progress_percentage = self.progress / self.total_timesteps
        self._update_progress_bar(progress_percentage)
        return True

    def _update_progress_bar(self, progress_percentage):
        bar_length = 30
        block = int(round(bar_length * progress_percentage))
        progress_bar = "[" + "=" * (block - 1) + ">" + "-" * (bar_length - block) + "]"
        output = (
            f"{self.jobname} progress: {progress_bar} {progress_percentage * 100:.2f}%"
        )
        print(output)


class AsyncProgressBarManager:
    def __init__(self):
        self.progress_bars = {}

    def get_progress_bar(self, job_id, timesteps=0):
        if job_id not in self.progress_bars:
            self.progress_bars[job_id] = ProgressBarCallback(timesteps, job_id)
        return self.progress_bars[job_id]


get_progress_bar = AsyncProgressBarManager().get_progress_bar
