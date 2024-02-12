"""

This module contains functions and classes for data ingestion and preprocessing.

Functions:
- process_data(file_path): Reads and preprocesses the data from a CSV file.
- get_cv_dfs(df, cv_periods): Splits the data into cross-validation sets.
- train_test_split(cv_dfs, i): Splits the cross-validation sets into train and test sets.

Classes:
- CVIngestionPipeline: A class for creating a cross-validation ingestion pipeline.

"""
from typing import Union
import pandas as pd
import numpy as np

used_cols = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ebitdamargin",
    "fcfps",
    "pb",
    "ps",
    "pe",
    "de",
    "capex",
    "revenue",
]
macro_cols = [
    "CPILFESL",
    "CPIENGSL",
    "TTLCONS",
    "JTSJOR",
    "JTSLDR",
    "JTSQUR",
    "JTSHIR",
    "GDP",
    "UNRATE",
    "CIVPART",
    "CES0500000003",
    "AWHAETP",
    "PAYEMS",
    "ISRATIO",
    "DGS10",
    "M2REAL",
    "BAA10Y",
    "DFF",
    "DEXJPUS",
    "DEXUSEU",
    "EURJPY",
    "DEXUSUK",
    "GBPJPY",
    "DEXSZUS",
    "CHFJPY",
    "INTGSTJPM193N",
    "IRSTCI01CHM156N",
    "IUDSOIA",
    "IR3TIB01DEM156N",
    "IRLTLT01DEM156N",
    "IRLTLT01FRM156N",
    "IRLTLT01ITM156N",
    "ECBASSETSW",
    "T10Y2Y",
    "JPNASSETS",
    "IRLTLT01JPM156N",
    "IRLTLT01GBM156N",
    "WTISPLC",
    "DEXCAUS",
]


def process_data(file_path):
    """
    Process the data from a CSV file and perform various transformations.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The processed DataFrame containing the transformed data.
    """
    df = (
        pd.read_csv(file_path, parse_dates=True)
        .set_index("ticker")
        .dropna(subset=["close", "datekey"])
        .reset_index()
        .set_index(["date", "ticker"])
        .sort_index()
    ).drop("dimension", axis=1)
    df = df.dropna(subset=["datekey"])
    df["spot"] = df["closeadj"]
    df["capexratio"] = df["capex"] / df["equity"]
    df = df.dropna(subset=["spot"])

    df = df[used_cols + macro_cols + ["spot"]]
    df["release_indicator"] = (
        (df.groupby(level="ticker")["revenue"].shift() != df["revenue"])
        .astype(int)
        .fillna(0)
    )
    df["intraday_vol"] = np.log(df["high"] / df["close"]) * np.log(
        df["high"] / df["open"]
    ) + np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"])

    return df


def get_cv_dfs(df, cv_periods):
    """
    Get cross-validation dataframes from a given dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    cv_periods (int): The number of cross-validation periods.

    Returns:
    list: A list of cross-validation dataframes.
    """
    cv_dfs = [
        df.loc[
            df.index.get_level_values("date").unique()[
                i
                * int(len(df.index.get_level_values("date").unique()) / cv_periods) : (
                    i + 1
                )
                * int(len(df.index.get_level_values("date").unique()) / cv_periods)
            ]
        ]
        for i in range(cv_periods)
    ]
    return cv_dfs


def train_test_split(cv_dfs, i):
    """
    Split the data into training and testing sets for cross-validation.

    Parameters:
    cv_dfs (list): List of dataframes for cross-validation.
    i (int): Index of the current fold.

    Returns:
    tuple: A tuple containing the training and testing dataframes.
    """
    train = pd.concat(cv_dfs[: i + 1])
    test = cv_dfs[i + 1]
    train_mean = train[used_cols + macro_cols].mean()
    train_std = train[used_cols + macro_cols].std()
    train[used_cols + macro_cols] = (
        train[used_cols + macro_cols] - train_mean
    ) / train_std
    test[used_cols + macro_cols] = (
        test[used_cols + macro_cols] - train_mean
    ) / train_std
    return train, test


class CVIngestionPipeline:
    """
    A cross-validation ingestion pipeline for trading data.

    Parameters:
    - file_path (str): The path to the data file.
    - cv_periods (int): The number of cross-validation periods.
    - cv_period_length (int): The length of each cross-validation period.
    - start_date (str or None): The start date for data processing.
      If None, all data is processed.

    Attributes:
    - df (DataFrame): The processed data.
    - cv_periods (int): The number of cross-validation periods.
    - cv_dfs (list): A list of cross-validation dataframes.
    - i (int): The current iteration index.

    Methods:
    - __iter__(): Returns an iterator for cross-validation train-test splits.
    - __len__(): Returns the number of cross-validation periods.

    Usage:
    ```
    pipeline = CVIngestionPipeline(file_path, cv_periods=5, cv_period_length=0,
      start_date=None)
    for train, test in pipeline:
        # Perform cross-validation training and testing
    ```
    """

    def __init__(
        self,
        file_path,
        cv_periods=5,
        cv_period_length=0,
        start_date: Union[str, None] = None,
    ):
        if start_date:
            self.df = process_data(file_path)
        if cv_period_length:
            self.cv_periods = int(
                len(self.df.index.get_level_values("date").unique()) // cv_period_length
            )
        else:
            self.cv_periods = cv_periods
        self.cv_dfs = get_cv_dfs(self.df, self.cv_periods)
        self.i = 0

    def __iter__(self):
        if self.i == self.cv_periods - 1:
            return
        train, test = train_test_split(self.cv_dfs, self.i)
        yield train, test
        self.i += 1

    def __len__(self):
        return self.cv_periods - 1
