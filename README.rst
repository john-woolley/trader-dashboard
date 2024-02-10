trader-dashboard
=============================

.. start-badges

.. list-table::
    :widths: 15 85
    :stub-columns: 1

    * - Build
      - | |Pylint|
    * - Package
      - | |Code style ruff|

.. |Pylint| image:: https://github.com/john-woolley/trader-dashboard/actions/workflows/pylint.yml/badge.svg?branch=main
   :target: https://github.com/john-woolley/trader-dashboard/actions/workflows/pylint.yml
.. |Code style ruff| image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
    :target: https://docs.astral.sh/ruff/

.. end-badges

Reinforcement Learning Dashboard with Sanic and ReactPy

Introduction
------------
This project aims to create a dashboard for reinforcement learning in trading using Sanic and ReactPy. It provides a user-friendly interface for training, monitoring and analyzing trading strategies.

## Features
- Real-time data visualization
- Interactive charts
- Performance metrics tracking
- Strategy comparison

Installation
------------
To install and run the dashboard, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/john-woolley/trader-dashboard.git
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Start the Sanic server:
    ```
    python app.py
    ```

4. Open the dashboard in your web browser:
    ```
    http://localhost:8000
    ```

Usage
-----
Once the dashboard is up and running, you can perform the following actions:

- Define a reinforcement model based on your own provided market data and free data APIs
- Train the model using vectorized environments and track the learning curve and other metrics in real-time.
- Backtest trading strategies with cross-validation slices.
- Visualize trading strategy performance over time with interactive charts.