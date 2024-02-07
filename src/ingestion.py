import pandas as pd
import numpy as np 
from typing import Union

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
    "capexratio",
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
    df = (
        pd.read_csv(file_path, parse_dates=True)
        .set_index("ticker")
        .dropna(subset=["close", "datekey"])
        .reset_index()
        .set_index(["date", "ticker"])
        .sort_index()
    ).drop("dimension", axis=1)
    df = df.dropna(subset=['datekey'])
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
    train = pd.concat(cv_dfs[:i + 1])
    test = cv_dfs[i + 1]
    train_mean = train[used_cols + macro_cols].mean()
    train_std = train[used_cols + macro_cols].std()
    train[used_cols + macro_cols] = (train[used_cols + macro_cols] - train_mean) / train_std
    test[used_cols + macro_cols] = (test[used_cols + macro_cols] - train_mean) / train_std
    return train, test

class CVIngestionPipeline:
    def __init__(self, file_path, cv_periods=5, cv_period_length=0, start_date: Union[str, None] = None):
        if start_date:
            self.df = process_data(file_path).loc[start_date:]
        if cv_period_length:
            self.cv_periods = int(len(self.df.index.get_level_values("date").unique()) / cv_period_length)
        else:
            self.cv_periods = cv_periods
        self.cv_dfs = get_cv_dfs(self.df, self.cv_periods)
        self.i = 0

    def __iter__(self):
        if self.i == self.cv_periods - 1:
            raise StopIteration
        train, test = train_test_split(self.cv_dfs, self.i)
        yield train, test
        self.i += 1

    def __len__(self):
        return self.cv_periods - 1