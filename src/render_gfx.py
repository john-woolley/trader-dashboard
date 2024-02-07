import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from scipy import stats

def moving_average(values: np.ndarray, window: int):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


x, y = ts2xy(load_results('log/252_days/SAC'), "timesteps")
y = moving_average(y, window=1000)
x = x[-len(y):]
zipped = list(zip(x, y))
df = pd.DataFrame(zipped, columns=["timesteps", "rewards"])
df = df.set_index("timesteps")
# df = df.clip(lower=-200)
fig = plt.figure('Learning Curve Smoothed')
ax = fig.add_subplot(111)
ax.plot(df.index, df["rewards"], label="Smoothed Rewards")
plt.title("Learning Curve Smoothed")
plt.show()
plt.savefig("test_render_252d.png")


# render_df = pd.read_csv("td3_stock_allocator_252d4.csv")
# df = render_df.set_index("Date")
# net_leverage = np.sum(df.filter(like="leverage"), axis=1)
# gross_leverage = np.sum(np.abs(df.filter(like="leverage")), axis=1)
# df["net_leverage"] = net_leverage
# df["gross_leverage"] = gross_leverage
# fig, ax = plt.subplots(figsize=(18, 6))
# df.plot(y="market_value", use_index=True, ax=ax, color="lightgrey", secondary_y=True)
# df.plot(y="gross_leverage", use_index=True, ax=ax, color="blue")
# df.plot(y="net_leverage", use_index=True, ax=ax, color="red")
# plt.title("Portfolio Leverage")
# plt.savefig(f"test_render_252d_td4.png")
