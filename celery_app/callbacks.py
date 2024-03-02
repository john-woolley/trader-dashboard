from stable_baselines3.common.callbacks import BaseCallback

class ProgressBarCallback(BaseCallback):
    """
    Callback for displaying a slick ASCII progress bar with a '>' indicator.
    """
    def __init__(self, total_timesteps, jobname, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress = 0
        self.jobname = jobname

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            self.progress += 100
            progress_percentage = self.progress / self.total_timesteps
            self._update_progress_bar(progress_percentage)
        return True

    def _update_progress_bar(self, progress_percentage):
        bar_length = 30
        block = int(round(bar_length * progress_percentage))
        progress_bar = "[" + "=" * (block - 1) + ">" + "-" * (bar_length - block) + "]"
        print(f"{self.jobname} progress: {progress_bar} {progress_percentage * 100:.2f}%", end="", flush=True)