#############################
trader-dashboard
#############################


.. image:: https://github.com/john-woolley/trader-dashboard/assets/53134776/d0ba45c3-ddff-4cb6-a6b8-9cb8c1b3b4b5
    :alt: trader-dashboard


.. start-badges

.. list-table::
    :widths: 15 85
    :stub-columns: 1

    * - Package
      - | |Supported implementations| |PyPI version| |Code style ruff|

.. |Code style ruff| image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
   :target: https://docs.astral.sh/ruff/
.. |PyPI version| image:: https://img.shields.io/pypi/pyversions/sanic.svg
   :alt: CPython
.. |Supported implementations| image:: https://img.shields.io/pypi/implementation/sanic.svg
    :alt: 3.8, 3.9, 3.10, 3.11
.. end-badges

=======================================================
Reinforcement Learning Dashboard with Sanic and Celery
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
- [x] Create a Celery worker
- [x] Create a DB API 
- [.] Create a backend API 
- [.] Implement critical sections in Rust
- [.] Create a DAG api for customized feature engineering
- [ ] Create a ReactPy front-end


*****************
API Specification
*****************

The API specification is a work in progress.  It will be updated as the project progresses.

- [x] Upload CSV data to data store: `upload_csv`
- [x] Standardize and partition data for time series cross validation: `prepare_data`
- [x] Start training and validation loop over the cross validation periods: `start_training`
- [ ] Report job status and progress: `get_jobs`
- [ ]  

***********
Screenshots
***********

.. image:: https://github.com/john-woolley/trader-dashboard/assets/53134776/1bf27317-2c14-4979-8734-08047370849f


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
