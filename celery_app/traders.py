import numpy as np
import gymnasium as gym
from celery_app import Trader as TraderRust


class Trader(gym.Env):
    def __init__(
        self,
        table_name: str,
        jobname: str,
        chunk: int,
        initial_balance: float = 1e5,
        n_lags=10,
        transaction_cost=0.0025,
        ep_length=252,
        test=False,
        risk_aversion=0.9,
        render_mode="none",
    ):
        """
        table_name: String,
        jobname: String,
        chunk: usize,
        initial_balance: f64,
        n_lags: usize,
        transaction_cost: f64,
        ep_length: usize,
        test: bool,
        risk_aversion: f64,
        render_mode: String,
        """
        self.trader = TraderRust(
            table_name,
            jobname,
            chunk,
            initial_balance,
            n_lags,
            transaction_cost,
            ep_length,
            test,
            risk_aversion,
            render_mode,
        )
        self.no_symbols = self.trader.no_symbols
        low = [0] * self.no_symbols + [0.001, 0.6, -1]
        high = [1] * self.no_symbols + [0.1, 1.5, 1]
        self.action_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(self.no_symbols + 3,),
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lags, 43 * self.no_symbols + 39)
        )

    def step(self, action):
        return self.trader.step(action)

    def reset(self):
        return np.array(self.trader.reset())

    def render(self, mode="human"):
        return self.trader.render(mode)

if __name__ == '__main__':
    env = Trader("test_upload2", "test_job2", 0)
    print(env)