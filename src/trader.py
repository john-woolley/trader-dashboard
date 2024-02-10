# Description: Delta hedging environment for Soft Actor Critic (SAC) RL model

"""
This module contains the implementation of the Trader class, which represents
the delta hedging environment for the Soft Actor Critic (SAC) RL model. The
Trader class is a subclass of the gym.Env class from the gymnasium library.

The Trader environment simulates a trading environment where an agent can
perform buy and sell actions on a set of assets. The environment provides
observations of the current state, which include lagged observations of asset
prices and other market indicators. The agent's goal is to maximize its
portfolio value by making optimal trading decisions.

The Trader class provides methods for initializing the environment, resetting
it to its initial state, and executing trades. It also calculates various
metrics such as portfolio value, position weights, and rewards. The class
implements the gym.Env interface, which allows it to be used with reinforcement
learning algorithms.

Example usage:

# Create a Trader environment
data = load_data()  # Load market data
trader = Trader(data)

# Reset the environment
observation, info = trader.reset()

# Execute a trade
amount = 1000  # Trade amount
sign = 1  # Buy
underlying = 0  # Index of the underlying asset
trader._trade(amount, sign, underlying)

# Get the current portfolio value
portfolio_value = trader.current_portfolio_value

# Get the position weights
position_weights = trader.net_position_weights

# Get the reward for the current step
reward = trader._get_reward()

"""

import random
from typing import Tuple, Any, SupportsFloat, Dict
from collections import deque

import numpy as np
import gymnasium as gym
import pandas as pd

from gymnasium import spaces
from scipy.stats import yeojohnson
from sanic.log import logger

from src.ingestion import macro_cols, used_cols


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
    return (m > val).argmax(axis) / m.shape[axis]


class Trader(gym.Env):
    """
    The Trader class represents an environment for trading simulation.

    Parameters:
    - data (dict[pd.DataFrame]): A dictionary of pandas DataFrames
        containing the data for each asset.
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

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 1e5,
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
        - data (dict[pd.DataFrame]): A dictionary of pandas DataFrames
          containing the data for each asset.
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
        self.initial_balance: float = initial_balance
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
        return self.dates[self.current_index]

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
        self.spot = self.data.loc[self.dates[self.period + self.current_step], "spot"]
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
        p = self.period
        s = self.current_step
        macro_len = len(macro_cols)
        curr_date = self.dates[p + s]
        if self.current_step < 10:
            prev_dates = self.dates[p : p + s + 1]
        else:
            prev_dates = self.dates[p + s - 10 : p + s + 1]
        spot = self.data.loc[curr_date, "spot"]
        spot_window = self.data.loc[prev_dates, "spot"].to_frame()
        assert isinstance(spot_window, pd.DataFrame)
        log_spot_window = np.log(spot_window)
        assert isinstance(log_spot_window, pd.DataFrame)
        if self.current_step > 0:
            spot_returns = log_spot_window.unstack().diff().dropna().iloc[-1]
            spot_returns = spot_returns.values
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
        macro_state = [self.data.loc[curr_date, col] for col in macro_cols]
        macro_state = np.array(macro_state)
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
        std = np.std(self.return_series)
        assert isinstance(std, float)
        return std

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
        return np.array(self.state_buffer)

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
                f"Date: {self.dates[self.current_index]}, "
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
            "Date": [self.dates[self.current_index]],
            "market_value": [self.current_portfolio_value],
            "balance": [self.balance],
            "paid_slippage": [self.paid_slippage],
        }
        net_lev_dict = {
            f"leverage_{self.symbols[i]}": self.net_position_values[i]
            for i in range(self.no_symbols)
        }
        action_dict = {
            f"action_{self.symbols[i]}": self.action[i] for i in range(self.no_symbols)
        }
        state_dict = {**state_dict, **net_lev_dict, **action_dict}
        step_df = pd.DataFrame.from_dict(state_dict)
        dfs = [self.render_df, step_df]
        self.render_df = pd.concat(dfs, ignore_index=True)
