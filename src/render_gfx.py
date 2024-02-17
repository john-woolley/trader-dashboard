"""
This module contains functions for rendering graphics related to trading data.

Functions:
- moving_average(values: np.ndarray, window: int) -> np.ndarray:
  Smooths values by doing a moving average.
"""
import numpy as np
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from reactpy_apexcharts import ApexChart
from reactpy import component, html, run


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Smooth values by doing a moving average.

    Args:
        values (np.ndarray): The values to be smoothed.
        window (int): The size of the moving average window.

    Returns:
        np.ndarray: The smoothed values.
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


@component
def get_rewards_curve_figure(log_dir: str):
    """
    Get a figure of the rewards curve.

    Args:
        log_dir (str): The directory containing the log files.

    Returns:
        figure: The figure of the rewards curve.
    """
    x, y = ts2xy(load_results(log_dir), "timesteps")
    smoothed_y = moving_average(y, window=10)
    smoothed_x = x[-len(y) :]
    fig = ApexChart(
        options={
            "chart": {"id": "smoothed-rewards-curve"},
            "xaxis": {"category": smoothed_x.tolist()},
        },
        series=smoothed_y.tolist(),
        chart_type="line",
        width=800,
        height=400,
    )
    return html.div(fig)
