import pytest
from src.trader import Trader

# Load market data
data = "test_table"
chunk = 0


@pytest.fixture
def trader():
    return Trader(data, chunk, test=True)


def test_reset(trader):
    observation, _ = trader.reset()
    assert observation is not None


def test_trade(trader):
    amount = 1000  # Trade amount
    sign = 1  # Buy
    underlying = 0  # Index of the underlying asset
    trader._trade(amount, sign, underlying)
    assert trader.net_leverage[underlying] == 1000
