"""
Stock picker environemnt for Soft Actor Critic (SAC) RL model

This module defines the Trader class, which represents an environment for the
stock picker model. The environemnt contains price and fundamental data for
a particular industry as well as macroeconomic data. The environment is used to
train a Soft Actor Critic (SAC) RL model to make trading decisions based on the
observed data.

The environment is implemented using the gym library, which is
a toolkit for developing and comparing reinforcement learning algorithms.
The Trader class inherits from the gym.Env class and implements the necessary
methods required by the gym library for creating custom environments.

The observation space is a 2D array of shape (n_lags, stock_features * n 
+ macro_features), where n is the number of stocks and n_lags is the number of
lagged observations to include in the state buffer.

The action space is a 1D array of shape (n + 2), where n is the number of stocks.
The two additional elements in the action space are the gross short and long
leverage targets.

The stock picker uses a concept of a model portfolio to make trading decisions.
The model portfolio is a 1D array of shape (n), where n is the number of stocks.
The model portfolio represents the desired portfolio weights for each stock.
The difference between the model portfolio and the actual portfolio is used to
determine the trading actions up to the specified leverage limits and subject
to a turnover constraint.

The reward is calculated based using the risk-adjusted return of the portfolio.
The risk aversion parameter is used to adjust the reward based on the risk that
models a trader's subjective risk preference.

The calculation is based on the following formula:
reward = yeojohnson((return - risk_free_rate) / volatility, risk_aversion)

 
Example usage:
    env = Trader(table_name='stock_data', chunk=0, initial_balance=100000)
    observation, info = env.reset()
    action = env.action_space.sample()
    next_observation, reward, done, info = env.step(action)

"""

import random
from typing import Tuple, Any, SupportsFloat, Dict
from collections import deque

import numpy as np
import gymnasium as gym
import pandas as pd
import polars as pl

from gymnasium import spaces
from scipy.stats import yeojohnson
from sanic.log import logger

from src.ingestion import macro_cols, used_cols

from src import db


def get_percentile(val, m, axis=0):
    """
    Calculate the percentile of a value in an array along a specified axis.

    Parameters:
    val (float): The value for which the percentile is calculated.
    m (numpy.ndarray): The input array.
    axis (int, optional): The axis along which the percentile is calculated.
    Default is 0.

    Returns:
    float: The percentile of the value in the array.

    """
    val = val.reshape(1, -1)
    return np.sum(m > val, axis=axis) / m.shape[axis]


def get_dt_expr(dates: np.ndarray) -> pl.Expr:
    """
    Get a polars expression for a date column.

    Parameters:
    dates (np.ndarray): The input array of dates.

    Returns:
    pl.Expr: The polars expression for the date column.

    """
    return dates.cast(pl.Date)


class Trader(gym.Env):
    """
    The Trader class represents an environment for the stock picker model.
    """

    def __init__(
        self,
        table_name: str,
        chunk: int,
        initial_balance: float = 1e5,
        n_lags=10,
        transaction_cost=0.0025,
        ep_length=252,
        test=False,
        risk_aversion=0.9,
        render_mode=None,
    ) -> None:
        """
        Initializes the Trader environment.

        Parameters:
        - table_name (str): The name of the table containing the data.
        - chunk (int): The chunk number for the data.
        - initial_balance (float): The initial balance of the trader.
            Default is 100000.
        - n_lags (int): The number of lagged observations to include in the
            state. Default is 10.
        - transaction_cost (float): The transaction cost as a percentage of
            the traded amount. Default is 0.0025.
        - ep_length (int): The length of each episode in trading steps.
            Default is 252.
        - test (bool): Whether the environment is in test mode or not.
            Default is False.
        - risk_aversion (float): The risk aversion parameter for calculating
            the reward. Default is 0.9.
        - render_mode (str): The rendering mode for visualization. Default is None.
        """
        super().__init__()
        self.render_mode = render_mode
        self.risk_aversion = risk_aversion
        self.data = db.read_std_cv_table(table_name, chunk).sort("date")
        self.data = self.data.with_columns(
            pl.col("date").cast(pl.Date).alias("date"),
            pl.col("capex").truediv(pl.col("equity")).alias("capexratio"),
            pl.col("closeadj").alias("spot"),
        ).fill_nan(0.0)
        self.dates = self.data.select("date").unique().collect()
        self.symbols = self.data.select("ticker").unique().collect()
        self.no_symbols = len(self.symbols)
        self.test = test
        self.ep_length = ep_length
        if self.test:
            self.ep_length = len(self.dates)
        self.initial_balance: float = initial_balance
        low = [-1] * self.no_symbols + [0.05, 1.0]
        high = [1] * self.no_symbols + [0.25, 1.5]
        self.action_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(self.no_symbols + 2,),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lags, 17 * self.no_symbols + 39)
        )
        self.transaction_cost = transaction_cost
        self.current_step = 0
        self.buffer_len = n_lags
        self.render_df = pl.LazyFrame()
        self.render_dict: Dict[int, Dict[str, Any]] = {}
        self.balance = self.initial_balance
        self.net_leverage = np.zeros(self.no_symbols)
        self.model_portfolio = np.zeros(self.no_symbols)
        self.spot = np.zeros(self.no_symbols)
        self.prev_spot = np.zeros(self.no_symbols)
        self.vol_ests = np.array([0.1] * self.no_symbols)
        self.return_series = np.array([])
        self.var = 0
        self.log_ret = 0
        self.state_buffer: deque = deque([], self.buffer_len)
        self.paid_slippage = 0
        self.short_leverage = 0.05
        self.long_leverage = 1.0
        self.rate = 0.01
        self.prev_portfolio_value = 0
        self.period = 0
        self.action = np.zeros(self.no_symbols)

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
        return self.net_leverage * self.spot

    @property
    def total_net_position_value(self):
        """
        Calculates the hedge value.

        Returns:
        float: The hedge value.
        """
        return np.dot(self.net_leverage, self.spot)

    @property
    def total_gross_position_value(self):
        """
        Calculates the gross hedge value.

        Returns:
        float: The gross hedge value.
        """
        return np.dot(np.abs(self.net_leverage), self.spot)

    @property
    def current_index(self):
        """
        Returns the current index by adding the period to the current step.
        """
        return self.period + self.current_step

    @property
    def current_date(self):
        """
        Returns the current date from the list of dates.

        Returns:
            str: The current date.
        """
        item = self.dates[self.current_index].item()
        return pl.datetime(item.year, item.month, item.day)

    @property
    def model_portfolio_value(self):
        """
        Calculates the total value of the model portfolio.

        Returns:
            float: The total value of the model portfolio.
        """
        return np.sum(self.model_portfolio)

    def _get_model_portfolio_weights(self):
        scaled = self.model_portfolio.copy()
        shorts = np.where(self.model_portfolio < 0)[0]
        longs = np.where(self.model_portfolio > 0)[0]
        short_weights = np.abs(self.model_portfolio[shorts])
        long_weights = self.model_portfolio[longs]
        sum_weight_short = np.sum(short_weights)
        sum_weight_long = np.sum(long_weights)
        scaled[shorts] = scaled[shorts] / sum_weight_short * self.short_leverage
        scaled[longs] = scaled[longs] / sum_weight_long * self.long_leverage
        return scaled

    def reset(self, seed=42):
        """
        Resets the environment to its initial state.

        Parameters:
        - seed (int): The seed value for random number generation.
          Default is 42.

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
            self.period = random.randint(0, len(self.dates) - self.ep_length)  # nosec
        self.spot = (
            self.data.select("spot", "date")
            .filter(pl.col("date") == self.current_date)
            .drop("date")
            .collect()
            .to_numpy()
            .flatten()
        )
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
        p = self.period
        s = self.current_step
        macro_len = len(macro_cols)
        if self.current_step < self.buffer_len:
            prev_dates = get_dt_expr(self.dates[p : p + s + 1])
        else:
            prev_dates = get_dt_expr(self.dates[p + s - self.buffer_len : p + s + 1])
        spot_window = (
            self.data.select("spot", "date", "ticker", *used_cols + macro_cols)
            .filter(pl.col("date").is_in(prev_dates))
            .drop("date", "ticker")
            .collect()
            .to_numpy()
            .reshape(len(prev_dates), -1, len(used_cols) + len(macro_cols) + 1)
        )
        spot = spot_window[-1, :, 0].reshape(-1)
        stock_state = spot_window[-1, :, 1:-macro_len].reshape(-1)
        macro_state = spot_window[-1, -1, -macro_len:]
        self.spot = spot
        log_spot_window = np.log(spot_window[:, :, 0])
        if self.current_step > 0:
            spot_returns = np.diff(log_spot_window, axis=0)[-1]
        else:
            spot_returns = np.zeros(self.no_symbols)
        spot_rank = get_percentile(spot, spot_window[:, :, 0], axis=0)
        stock_state = np.concatenate([stock_state, spot_rank, spot_returns])
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
        std = np.std(self.return_series) if len(self.return_series) > 0 else 0.05
        return std

    def _get_reward(self) -> np.ndarray:
        """
        Calculate the reward for the trader.

        Returns:
            float: The calculated reward.
        """
        ret = self._get_return()
        rf_ret = self.rate / 252
        ret_vol = self.return_volatility or 0.05
        reward = yeojohnson((ret - rf_ret) / ret_vol, self.risk_aversion)
        assert isinstance(reward, np.ndarray)
        return reward

    def _accrue_interest(self) -> None:
        """
        Accrues interest on the account balance based on the specified
        interest rate.

        Returns:
        None
        """
        self.balance *= 1.0 + (float(self.rate) / 252.0)

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

    def step(self, action) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Executes a single step in the trading environment.

        Args:
            action (list): A list of actions to be taken for each asset.

        Returns:
            tuple: A tuple containing the observation, reward, done flag,
            and additional info.
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
        deviations = (
            deviations
            / sum_deviation
            * np.min(
                [sum_deviation, max(self.balance / self.current_portfolio_value, 0.01)]
            )
        )
        for idx, deviation in enumerate(deviations):
            net = deviation * self.current_portfolio_value
            amount = net / self.spot[idx]
            sign = 1 if amount > 0 else -1
            self._trade(abs(amount), sign, underlying=idx)
        self.state_buffer.append(self._get_state_frame())
        self._update_return_series()
        reward = self._get_reward()
        self.action = action
        self.render(mode=str(self.render_mode))
        self.current_step += 1
        done = (
            self.current_step == self.ep_length - 1
            or self.current_portfolio_value < self.initial_balance * 0.5
        )
        if done:
            self.reset()
        info: Dict[str, Any] = {}
        obs = self._get_observation()
        return obs, reward, done, False, info

    def _get_observation(self) -> np.ndarray:
        """
        Get the observation for the current step.

        Returns:
            np.ndarray: The observation for the current step.
        """
        obs = np.array(self.state_buffer)
        assert obs.shape == (self.buffer_len, 17 * self.no_symbols + 39)
        return obs

    def render(self, mode: str = "human") -> None:
        """
        Renders the current state of the trader.

        Parameters:
        - mode (str): The rendering mode. Default is "human".

        Returns:
        None
        """

        if mode == "human":
            output = (
                f"Step:{self.current_step}, "
                f"Date: {pl.select(self.current_date.cast(pl.String)).item()}, "
                f"Market Value: {self.current_portfolio_value:.2f}, "
                f"Balance: {self.balance:.2f}, "
                f"Stock Owned: {self.total_net_position_value:.2f}, "
                f"Reward: {self._get_reward():.2f}, "
                f"Gross Leverage: {self.total_gross_position_value:.2f}"
            )
            logger.info(output)
        else:
            pass

        state_dict = {
            "Date": [pl.select(self.current_date.cast(pl.String)).item()],
            "market_value": [self.current_portfolio_value],
            "balance": [self.balance],
            "paid_slippage": [self.paid_slippage],
        }
        net_lev_dict = {
            f"leverage_{self.symbols[i].item()}": self.net_position_values[i]
            for i in range(self.no_symbols)
        }
        action_dict = {
            f"action_{self.symbols[i].item()}": self.action[i]
            for i in range(self.no_symbols)
        }
        state_dict.update(net_lev_dict)
        state_dict.update(action_dict)

        self.render_dict.update({self.current_step: state_dict})

    def get_render(self) -> pd.DataFrame:
        """
        Get the render dataframe.

        Returns:
            pd.DataFrame: The render dataframe.
        """
        return pd.DataFrame(self.render_dict).T
