from stable_baselines3.common.callbacks import BaseCallback
from cache import cache


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
