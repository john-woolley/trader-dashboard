#############################
trader-dashboard
#############################


.. image:: https://github.com/john-woolley/trader-dashboard/assets/53134776/d0ba45c3-ddff-4cb6-a6b8-9cb8c1b3b4b5
    :alt: trader-dashboard


.. start-badges

.. list-table::
    :widths: 15 85
    :stub-columns: 1

    * - Build
      - | |Tests| |Pylint| 
    * - Package
      - | |Supported implementations| |PyPI version| |Code style ruff|

.. |Pylint| image:: https://github.com/john-woolley/trader-dashboard/actions/workflows/pylint.yml/badge.svg?branch=main
   :target: https://github.com/john-woolley/trader-dashboard/actions/workflows/pylint.yml
.. |Code style ruff| image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
   :target: https://docs.astral.sh/ruff/
.. |PyPI version| image:: https://img.shields.io/pypi/pyversions/sanic.svg
   :alt: CPython
.. |Tests| image:: https://github.com/john-woolley/trader-dashboard/actions/workflows/tests.yml/badge.svg?branch=main
   :target: https://github.com/john-woolley/trader-dashboard/actions/workflows/tests.yml
.. |Supported implementations| image:: https://img.shields.io/pypi/implementation/sanic.svg
    :alt: 3.8, 3.9, 3.10, 3.11
.. end-badges

=======================================================
Reinforcement Learning Dashboard with Sanic and ReactPy
=======================================================


************
Introduction
************

NOTE: This project is an actively developed work in progress.  If you are seeing this, please know that it is not yet ready for use.  Please check back later.

This project aims to create a dashboard for reinforcement learning in trading using Sanic and ReactPy. It provides a user-friendly interface for training, monitoring and analyzing trading strategies.


**********
Milestones
**********

- [x] Create reinforcement training environment for asset allocation
- [x] Create a Sanic server
- [.] Create a backend API 
- [ ] Create a ReactPy front-end

********
Features
********

- Real-time data visualization
- Interactive charts
- Performance metrics tracking
- Strategy comparison

************
Installation
************

To install and run the dashboard, follow these steps.

^^^^^^^^^^^^^^^^^^^^^^
1. Install PostgresSQL
^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    sudo apt-get update
    sudo apt-get install postgresql postgresql-contrib

    # or on Fedora

    sudo dnf install postgresql-server postgresql-contrib


^^^^^^^^^^^^^^^^^^^^^^^
1. Clone the repository
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    git clone https://github.com/john-woolley/trader-dashboard.git

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2. Install the required dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    pip install -r requirements.txt

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
3. Start the Sanic server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    python app.py

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
4. Open the dashboard in your web browser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    http://localhost:8000

******
Usage
******

Once the dashboard is up and running, you can perform the following actions:

- Define a reinforcement model based on your own provided market data and free data APIs
- Train the model using vectorized environments and track the learning curve and other metrics in real-time.
- Backtest trading strategies with cross-validation slices.
- Visualize trading strategy performance over time with interactive charts.
