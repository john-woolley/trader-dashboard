import pytest
from src.trader import Trader
import pandas as pd

# Load market data
data = pd.read_csv("tests/data.csv", parse_dates=True, index_col=["date", "ticker"])
data["spot"] = data["closeadj"]
data["capexratio"] = data["capex"] / data["equity"]


@pytest.fixture
def trader():
    return Trader(data, test=True)


def test_reset(trader):
    observation, _ = trader.reset()
    assert observation is not None


def test_trade(trader):
    amount = 1000  # Trade amount
    sign = 1  # Buy
    underlying = 0  # Index of the underlying asset
    trader._trade(amount, sign, underlying)
    assert trader.net_leverage[underlying] == 1000
