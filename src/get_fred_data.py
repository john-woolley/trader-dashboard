"""
This module provides functions and classes for retrieving and processing data from the FRED API.
"""
from typing import Union, List, Callable

import numpy as np
import pandas as pd
import pyfredapi as pf


API_KEY = "ab6fdf404949da4276d26b19b073de86"


def calc_pct_chg(series: pd.Series, freq=1):
    """
    Calculate the percentage change of a series.

    Args:
        series (pd.Series): The input series.
        freq (int, optional): The frequency of the percentage change
        calculation. Defaults to 1.

    Returns:
        pd.Series: The percentage change series.
    """
    return ((series / series.shift(freq)) - 1) * 100


def get_series(series_name: str, pct_chg=False, pct_chg_freq=1):
    """
    Get a series from the FRED API.

    Args:
        series_name (str): The name of the series.
        pct_chg (bool, optional): Whether to calculate the percentage change
        of the series. Defaults to False.
        pct_chg_freq (int, optional): The frequency of the percentage change
        calculation. Defaults to 1.

    Returns:
        pd.Series: The requested series.
    """
    series = (
        pf.get_series(series_id=series_name, api_key=API_KEY)
        .set_index("date")
        .value.rename(series_name)
    )
    if pct_chg:
        return calc_pct_chg(series, freq=pct_chg_freq)
    return series

class DataSet:
    """
    A class representing a dataset.

    Attributes:
        rid (int): The release ID of the dataset.
        series (list[str]): The list of series names in the dataset.
        pct_chg (Union[bool, list[bool]]): Whether to calculate the percentage change of the series.
        data (pd.DataFrame): The data of the dataset.
        release_dates (pd.DataFrame): The release dates of the dataset.
    """

    def __init__(
        self, rid: int, series: List[str], pct_chg: Union[bool, List[bool]] = False
    ):
        self.rid = rid
        self.series = series
        self.pct_chg = pct_chg
        self.data = self.get()
        self.data.index = pd.to_datetime(self.data.index)
        self.release_dates = self.get_release_dates()
        self.release_dates.index = pd.to_datetime(self.release_dates.index)
        self.get_release_offsets()

    def get(self):
        """
        Get the data of the dataset.

        Returns:
            pd.DataFrame: The data of the dataset.
        """
        data = {}
        if hasattr(self.pct_chg, "__iter__"):
            for i, series in enumerate(self.series):
                data[series] = get_series(series, pct_chg=self.pct_chg[i])
            return pd.DataFrame(data)
        for series in self.series:
            data[series] = get_series(series, pct_chg=self.pct_chg)
        return pd.DataFrame(data)

    def get_release_dates(self):
        """
        Get the release dates of the dataset.

        Returns:
            pd.DataFrame: The release dates of the dataset.
        """
        release_dates = pf.get_release_dates(
            release_id=self.rid, api_key=API_KEY)
        release_df = pd.DataFrame(
            release_dates["release_dates"]).set_index("date")
        return release_df

    def get_release_offsets(self):
        """
        Get the release offsets of the dataset.
        """
        self.data["release_date"] = pd.NaT
        for date in self.data.index:
            pred = self.release_dates.index[
                self.release_dates.index.get_indexer([date], method="bfill")[0]
            ]
            self.data.loc[date, "release_date"] = pred
        self.data.set_index("release_date", inplace=True)


class DataCollection:
    """
    A class representing a collection of datasets.

    Attributes:
        datasets (list[callable]): The list of dataset classes.
        data (pd.DataFrame): The merged data of all datasets.
        min_date (pd.Timestamp): The minimum date in the data.
        max_date (pd.Timestamp): The maximum date in the data.
        daterange (pd.DatetimeIndex): The date range of the data.
    """

    def __init__(self, datasets: List[Callable]):
        self.datasets = [dataset().get() for dataset in datasets]
        self.data = self.merge_datasets()
        self.min_date = self.data.index.min()
        self.max_date = self.data.index.max()
        self.daterange = pd.date_range(
            start=self.min_date, end=self.max_date, freq="D")
        self.data = self.data.groupby(level=0).first()
        self.data = self.data.reindex(self.daterange, fill_value=np.nan)
        self.data = self.data.ffill()
        self.data = self.data.dropna()

    def merge_datasets(self):
        """
        Merge the data of all datasets.

        Returns:
            pd.DataFrame: The merged data.
        """
        merged = self.datasets[0].data
        for dataset in self.datasets[1:]:
            merged = pd.merge(
                merged, dataset.data, left_index=True, right_index=True,
                how="outer"
            )
        return merged

    def get_data(self):
        """
        Get the data of the collection.

        Returns:
            pd.DataFrame: The data of the collection.
        """
        return self.data


class DefinedDataSet:
    """
    A base class for defined datasets.

    Attributes:
        rid (int): The release ID of the dataset.
        date_file_offset (int): The date file offset of the dataset.
        series (list[str]): The list of series names in the dataset.
        pct_chg (bool): Whether to calculate the percentage change of the series.
    """

    rid = 0
    date_file_offset = 0
    series = []
    pct_chg = False

    @classmethod
    def get(cls):
        """
        Get the dataset.

        Returns:
            DataSet: The dataset.
        """
        return DataSet(cls.rid, cls.series, cls.pct_chg)

    @classmethod
    def get_release_dates(cls):
        """
        Get the release dates of the dataset.

        Returns:
            pd.DataFrame: The release dates of the dataset.
        """
        return pf.get_release_dates(release_id=cls.rid, api_key=API_KEY)


class CPIData(DefinedDataSet):
    """
    Represents a dataset for Consumer Price Index (CPI) data.
    
    Attributes:
        rid (int): The ID of the dataset.
        date_file_offset (int): The offset of the date file.
        series (list): A list of series names.
        pct_chg (bool): Indicates whether the data represents percentage change.
    """
    rid = 10
    date_file_offset = 40
    series = ["CPILFESL", "CPIENGSL"]
    pct_chg = True


class FOMCData(DefinedDataSet):
    """
    Represents a dataset containing FOMC (Federal Open Market Committee) data.
    """

    rid = 326
    date_file_offset = 41
    series = [
        "FEDTARMD",
        "FEDTARMDLR",
        "FEDTARRL",
        "FEDTARRM",
        "FEDTARCTM",
        "FEDTARCTH",
        "FEDTARRHLR",
        "FEDTARRH",
        "FEDTARCTLLR",
    ]
    pct_chg = False


class CONSTRData(DefinedDataSet):
    """
    A class representing Construction dataset.

    Attributes:
        rid (int): The identifier for the dataset.
        data_file_offset (int): The offset of the data file.
        series (list): The list of series names.
        pct_chg (bool): Indicates whether the data should be percentage change
        or not.
    """
    rid = 229
    data_file_offset = 40
    series = ["TTLCONS"]
    pct_chg = True


class JOLTSData(DefinedDataSet):
    """
    A class representing JOLTS (Job Openings and Labor Turnover Survey) data.

    Attributes:
        rid (int): The ID of the JOLTS data.
        date_file_offset (int): The offset of the date file.
        series (list): A list of series codes for JOLTS data.
        pct_chg (bool): Indicates whether percentage change is enabled or not.
    """
    rid = 192
    date_file_offset = 41
    series = ["JTSJOR", "JTSLDR", "JTSQUR", "JTSHIR"]
    pct_chg = False


class GDPData(DefinedDataSet):
    """
    Represents a dataset for GDP data.
    
    Attributes:
        rid (int): The ID of the dataset.
        date_file_offset (int): The offset in days for the date file.
        series (list): A list of series names.
        pct_chg (bool): Indicates whether the data represents percentage change.
    """

    rid = 53
    date_file_offset = 60
    series = ["GDP"]
    pct_chg = True


class EMPLData(DefinedDataSet):
    """
    A class representing employment report data.

    Attributes:
        rid (int): The ID of the dataset.
        date_file_offset (int): The offset of the date file.
        series (list): A list of series names.
        pct_chg (list): A list indicating whether each series should be
        calculated as a percentage change.
    """

    rid = 50
    date_file_offset = 40
    series = ["UNRATE", "CIVPART", "CES0500000003", "AWHAETP", "PAYEMS"]
    pct_chg = [False, False, True, True, True]


class MANUData(DefinedDataSet):
    """
    A class representing Manufacturing Data.

    Attributes:
        rid (int): The ID of the MANUData.
        date_file_offset (int): The offset of the date file.
        series (list): A list of series names.
        pct_chg (bool): A flag indicating whether to calculate percentage
        change.
    """

    rid = 25
    date_file_offset = 56
    series = ["ISRATIO"]
    pct_chg = True

if __name__ == "__main__":
    data_list = [CPIData, CONSTRData, JOLTSData, GDPData, EMPLData, MANUData]
    data_collection = DataCollection(data_list)

    market_data = [
        real_m2 := get_series("M2REAL", pct_chg=True),
        corp_spread := get_series("BAA10Y"),
        fedfunds := get_series("DFF"),
        usdjpy := get_series("DEXJPUS"),
        eurusd := get_series("DEXUSEU"),
        eurjpy := (usdjpy * eurusd).rename("EURJPY"),
        gbpusd := get_series("DEXUSUK"),
        gbpjpy := (usdjpy * gbpusd).rename("GBPJPY"),
        usdchf := get_series("DEXSZUS"),
        chfjpy := (usdjpy / usdchf).rename("CHFJPY"),
        jpy_on := get_series("INTGSTJPM193N"),  # JGBills
        chf_on := get_series("IRSTCI01CHM156N"),
        sonia := get_series("IUDSOIA"),
        eur_on := get_series("IR3TIB01DEM156N"),
        ger_10 := get_series("IRLTLT01DEM156N"),
        fra_10 := get_series("IRLTLT01FRM156N"),
        ita_10 := get_series("IRLTLT01ITM156N"),
        ecb_bs := get_series("ECBASSETSW", pct_chg=True, pct_chg_freq=4),
        crv_2_10 := get_series("T10Y2Y"),
        boj_assets := get_series("JPNASSETS", pct_chg=True),
        jgb_10 := get_series("IRLTLT01JPM156N"),
        gbp_10 := get_series("IRLTLT01GBM156N"),
        wti_spliced := get_series("WTISPLC"),
        usdcad := get_series("DEXCAUS"),
    ]
    cmt_10y = get_series("DGS10")

    market_df = cmt_10y.to_frame()
    for s in market_data:
        market_df = market_df.merge(
            s, how="outer", left_index=True, right_index=True)
    print(market_df)

    master_df = pd.merge(
        data_collection.data, market_df, how="outer",
        left_index=True, right_index=True
    ).ffill()
    master_df.to_csv("data/macro.csv")
