# Description: Delta hedging environment for Soft Actor Critic (SAC) reinforcement learning
from src.ingestion import macro_cols, used_cols, CVIngestionPipeline
import numpy as np
import gymnasium as gym
import random
from gymnasium import spaces
from collections import deque
import pandas as pd
from stable_baselines3 import SAC, PPO
from functools import partial
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import pandas as pd
from scipy.stats import yeojohnson
from typing import Union


def get_percentile(val, M, axis=0):
    return (M > val).argmax(axis) / M.shape[axis]


class Trader(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance=100000,
        n_lags=10,
        transaction_cost=0.0025,
        ep_length=252,
        test=False,
        risk_aversion=0.9,
        render_mode=None,
    ):
        """
        Initializes the Trader environment.

        Parameters:
        - data (dict[pd.DataFrame]): A dictionary of pandas DataFrames containing the data for each asset.
        - initial_balance (float): The initial balance of the trader. Default is 100000.
        - n_lags (int): The number of lagged observations to include in the state. Default is 10.
        - transaction_cost (float): The transaction cost as a percentage of the traded amount. Default is 0.0025.
        - ep_length (int): The length of each episode in trading steps. Default is 252.
        - test (bool): Whether the environment is in test mode or not. Default is False.
        """
        super().__init__()
        self.render_mode = render_mode
        self.risk_aversion = risk_aversion
        self.data = data
        self.dates = np.array(self.data.index.get_level_values(0).unique())
        self.symbols = np.array(self.data.index.get_level_values(1).unique())
        self.no_symbols = len(self.symbols)
        self.test = test
        self.ep_length = ep_length
        if self.test:
            self.ep_length = len(self.dates)
        self.initial_balance = initial_balance
        low = [-1] * self.no_symbols + [0.05, 1.0]
        high = [1] * self.no_symbols + [0.25, 1.5]
        self.action_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(self.no_symbols + 2,),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lags, 17 * self.no_symbols + 43)
        )
        self.transaction_cost = transaction_cost
        self.current_step = 0
        self.buffer_len = n_lags
        self.render_df = pd.DataFrame()
        self.balance = 0
        self.net_leverage = np.zeros(self.no_symbols)
        self.model_portfolio = np.zeros(self.no_symbols)
        self.spot = np.zeros(self.no_symbols)

    @property
    def current_portfolio_value(self):
        """
        Calculates the current portfolio value.

        Returns:
        float: The current portfolio value.
        """
        return self.balance + self.total_net_position_value

    @property
    def net_position_weights(self) -> np.ndarray:
        """
        Calculates the position weights.

        Returns:
        np.ndarray: The position weights.
        """
        return self.net_position_values / self.current_portfolio_value

    @property
    def net_position_values(self) -> np.ndarray:
        """
        Calculates the position value.

        Returns:
        np.ndarray: The position value.
        """
        try:
            assert isinstance(self.spot, np.ndarray)
            return self.net_leverage * self.spot
        except Exception as e:
            print(self.dates[self.current_index])
            print(self.spot)
            print(self.net_leverage)
            raise e

    @property
    def total_net_position_value(self):
        """
        Calculates the hedge value.

        Returns:
        float: The hedge value.
        """
        assert isinstance(self.spot, np.ndarray)
        return np.dot(self.net_leverage, self.spot)

    @property
    def total_gross_position_value(self):
        """
        Calculates the gross hedge value.

        Returns:
        float: The gross hedge value.
        """
        assert isinstance(self.spot, np.ndarray)
        return np.dot(np.abs(self.net_leverage), self.spot)

    @property
    def current_index(self):
        return self.period + self.current_step

    @property
    def current_date(self):
        return self.dates[self.current_index]

    @property
    def model_portfolio_value(self):
        return np.sum(self.model_portfolio)

    def _get_model_portfolio_weights(self):
        scaled = self.model_portfolio.copy()
        shorts = np.where(self.model_portfolio < 0)[0]
        longs = np.where(self.model_portfolio > 0)[0]
        short_weights = np.abs(self.model_portfolio[shorts])
        long_weights = self.model_portfolio[longs]
        sum_weight_short = np.sum(short_weights)
        sum_weight_long = np.sum(long_weights)
        scaled[shorts] = scaled[shorts] / \
            sum_weight_short * self.short_leverage
        scaled[longs] = scaled[longs] / sum_weight_long * self.long_leverage
        return scaled

    def reset(self, seed=42):
        """
        Resets the environment to its initial state.

        Parameters:
        - seed (int): The seed value for random number generation. Default is 42.

        Returns:
        - observation (object): The initial observation of the environment.
        - info (dict): An empty dictionary.
        """
        self.vol_ests = np.array([0.1] * self.no_symbols)
        self.return_series = np.array([])
        self.var = 0
        self.net_leverage = np.zeros(self.no_symbols)
        self.current_step = 0
        if self.test:
            self.period = 0
        else:
            self.period = random.randint(0, len(self.dates) - self.ep_length)
        self.spot = self.data.loc[
            self.dates[self.period + self.current_step], "spot"
        ]
        assert isinstance(self.spot, pd.Series)
        self.spot = self.spot.values
        self.balance = self.initial_balance
        self.log_ret = 0
        self.state_buffer = deque([], self.buffer_len)
        self.paid_slippage = 0
        self.model_portfolio = np.zeros(self.no_symbols)
        state_frame = self._get_state_frame()
        while len(self.state_buffer) < self.buffer_len:
            self.state_buffer.append(state_frame)
        return self._get_observation(), {}

    def _get_state_frame(self) -> np.ndarray:
        """
        Get the state frame for the current step.

        Returns:
            int: The predicted state for the current step.
        """
        macro_len = len(macro_cols)
        curr_date = self.dates[self.period + self.current_step]
        if self.current_step < 10:
            prev_dates = self.dates[self.period:self.period +
                                    self.current_step+1]
        else:
            prev_dates = self.dates[self.period +
                                    self.current_step-10: self.period+self.current_step + 1]
        spot = self.data.loc[curr_date, "spot"]
        spot_window = self.data.loc[prev_dates, "spot"].to_frame()
        assert isinstance(spot_window, pd.DataFrame)
        log_spot_window = np.log(spot_window)
        assert isinstance(log_spot_window, pd.DataFrame)
        if self.current_step > 0:
            spot_returns = log_spot_window.unstack(
            ).diff().dropna().iloc[-1].values
        else:
            spot_returns = np.zeros(self.no_symbols)
        assert isinstance(spot_returns, np.ndarray)
        assert isinstance(spot, pd.Series)
        spot_window_vals = spot_window.values
        spot_values = spot.values
        assert isinstance(spot_window_vals, np.ndarray)
        assert isinstance(spot_values, np.ndarray)
        spot_rank = get_percentile(spot_values, spot_window_vals, axis=0)
        assert isinstance(spot, pd.Series)
        self.spot = spot.values
        assert isinstance(self.data, pd.DataFrame)
        slices = []
        for col in used_cols:
            this_slice = self.data.loc[curr_date, col]
            assert isinstance(this_slice, pd.Series)
            this_slice = this_slice.reindex(self.symbols, fill_value=0).values
            slices.append(this_slice)
        stock_state = np.concatenate(slices)
        stock_state = np.concatenate([stock_state, spot_rank, spot_returns])
        macro_state = np.array([self.data.loc[curr_date, col]
                               for col in macro_cols])
        macro_state = np.concatenate(macro_state)[slice(None, None, macro_len)]
        state = np.concatenate(
            [stock_state, self.net_position_weights, self.model_portfolio]
        )
        state = np.concatenate([state, macro_state])

        return state

    def _trade(self, amount: float, sign: int, underlying: int) -> None:
        """
        Executes a trade of the given amount and sign.
        Sign should be 1 for buying and -1 for selling.

        Parameters:
        - amount (float): The amount to be traded.
        - sign (int): The sign of the trade (-1 for selling, 1 for buying).
        - underlying (str): The symbol of the underlying asset.

        Returns:
        None
        """
        assert isinstance(self.spot, np.ndarray)
        paid_slippage = amount * self.spot[underlying] * self.transaction_cost
        self.net_leverage[underlying] += amount * sign
        self.balance -= amount * sign * self.spot[underlying] + paid_slippage
        self.paid_slippage += paid_slippage

    def _update_return_series(self) -> None:
        """
        Updates the return series.

        Returns:
        None
        """
        ret = self._get_return() - self.rate / 252
        if ret < 0:
            self.return_series = np.append(self.return_series, ret)

    @property
    def return_volatility(self) -> float:
        """
        Calculates the return volatility.

        Returns:
        float: The return volatility.
        """
        assert isinstance(self.return_series, np.ndarray)
        try:
            std = np.std(self.return_series)
            assert isinstance(std, float)
            return std
        except:
            return 1

    def _get_reward(self) -> np.ndarray:
        """
        Calculate the reward for the trader.

        Returns:
            float: The calculated reward.
        """
        ret = self._get_return()
        rf_ret = self.rate / 252
        ret_vol = self.return_volatility or 1
        reward = yeojohnson((ret - rf_ret) / ret_vol, self.risk_aversion)
        assert isinstance(reward, np.ndarray)
        return reward

    def _accrue_interest(self) -> None:
        """
        Accrues interest on the account balance based on the specified interest rate.

        Returns:
        None
        """
        self.balance *= 1 + (self.rate / 252)

    def _get_return(self) -> float:
        """
        Calculate the return on investment.

        Returns:
            float: The return on investment as a decimal value.
        """
        return (
            np.log(self.current_portfolio_value / self.prev_portfolio_value)
            if self.current_portfolio_value > 0
            else -1
        )

    def _get_portfolio_weight(self, symbol: str) -> float:
        """
        Returns the portfolio weight for the given symbol.

        Parameters:
        - symbol (str): The symbol of the asset.

        Returns:
        float: The portfolio weight for the given symbol.
        """
        return self.net_position_values[symbol] / self.current_portfolio_value

    def step(self, action) -> tuple:
        """
        Executes a single step in the trading environment.

        Args:
            action (list): A list of actions to be taken for each asset.

        Returns:
            tuple: A tuple containing the observation, reward, done flag, and additional info.
        """
        assert isinstance(self.spot, np.ndarray)
        self.paid_slippage = 0
        self.prev_portfolio_value = self.current_portfolio_value
        self.prev_spot = self.spot
        self.rate = 0.01
        self._accrue_interest()
        for idx, x in enumerate(action[: self.no_symbols]):
            self.model_portfolio[idx] = x
        self.short_leverage = action[-2]
        self.long_leverage = action[-1]
        target_weights = self._get_model_portfolio_weights()
        deviations = np.zeros(self.no_symbols)
        for idx, x in enumerate(action[: self.no_symbols]):
            actual_weight = self.net_position_weights[idx]
            target_weight = target_weights[idx]
            deviations[idx] = target_weight - actual_weight
        sum_deviation = np.sum(np.abs(deviations))
        deviations = deviations / sum_deviation * \
            np.min([sum_deviation, max(self.balance /
                   self.current_portfolio_value, 0.01)])
        for idx, deviation in enumerate(deviations):
            net = deviation * self.current_portfolio_value
            amount = net / self.spot[idx]
            sign = 1 if amount > 0 else -1
            self._trade(abs(amount), sign, underlying=idx)
        self.state_buffer.append(self._get_state_frame())
        self._update_return_series()
        reward = self._get_reward()
        self.render(action, mode=self.render_mode)
        self.current_step += 1
        done = (
            self.current_step == self.ep_length - 1
            or self.current_portfolio_value < self.initial_balance * 0.5
        )
        if done:
            self.reset()
        info = {}
        obs = self._get_observation()
        return obs, reward, done, False, info

    def _get_observation(self):
        return self.state_buffer

    def render(self, action, mode: Union[str, None] = "human"):
        """
        Renders the current state of the trader.

        Parameters:
        - action (str): The action taken at the current step.
        - mode (str): The rendering mode. Default is "human".

        Returns:
        None
        """

        if mode == "human":
            print(
                (
                    f"Step:{self.current_step}, \
                      Date: {self.dates[self.current_index]}, \
                      Action: {action}, \
                      Positions: {self.net_leverage}, \
                      Market Value: {self.current_portfolio_value:.2f}, \
                      Balance: {self.balance:.2f}, \
                      Stock Owned: {self.total_net_position_value:.2f}, \
                        Reward: {self._get_reward():.2f} \
                      Gross Leverage: {self.total_gross_position_value:.2f} "
                )
            )
        else:
            pass
        state_dict = {
            "Date": [self.dates[self.current_index]],
            "market_value": [self.current_portfolio_value],
            "balance": [self.balance],
            "paid_slippage": [self.paid_slippage],
        }
        net_lev_dict = {
            f"leverage_{self.symbols[i]}": self.net_position_values[i]
            for i in range(self.no_symbols)
        }
        state_dict = {**state_dict, **net_lev_dict}
        step_df = pd.DataFrame.from_dict(state_dict)
        self.render_df = pd.concat(
            [self.render_df, step_df], ignore_index=True)


if __name__ == "__main__":
    model_name = "SAC"
    model = locals()[model_name]
    ep_length = 252
    cv_periods = 20
    train_start_date = "2019-04-03"
    data = CVIngestionPipeline(
        "data/master.csv", cv_periods, start_date=train_start_date)
    for i in range(0, len(data)):
        train, test = tuple(*iter(data))
        mfile = f"stock_allocator_{ep_length}d"
        log_dir = f"log/{ep_length}_days/{model_name}"
        env = SubprocVecEnv([partial(Trader, train, test=True)
                            for _ in range(64)])
        env = VecMonitor(env, log_dir)
        policy_kwargs = dict(
            net_arch=dict(pi=[4096, 2048, 1024, 1024], vf=[
                          4096, 2048,  1024, 1024], qf=[4096, 2048, 1024, 1024])
        )
        model_train = model(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            batch_size=128,
            use_sde=True,
        )
        model_train.learn(total_timesteps=500, progress_bar=True)
        model_train.save(f"{model_name}_stock_allocator_{ep_length}d_{i}")
        env_test = Trader(test, test=True)
        model_test = model.load(
            f"{model_name}_stock_allocator_{ep_length}d_{i}", env=env_test
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
        env_test.render_df.to_csv(
            f"{model_name}_stock_allocator_{ep_length}d_{i}.csv")
