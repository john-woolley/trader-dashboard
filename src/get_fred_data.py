import numpy as np
import pandas as pd
import pyfredapi as pf
from typing import Union

api_key = "ab6fdf404949da4276d26b19b073de86"


def calc_pct_chg(series: pd.Series, freq=1):
    return ((series / series.shift(freq)) - 1) * 100


def get_series(series_name: str, pct_chg=False, pct_chg_freq=1):
    series = (
        pf.get_series(series_id=series_name, api_key=api_key)
        .set_index("date")
        .value.rename(series_name)
    )
    if pct_chg:
        return calc_pct_chg(series, freq=pct_chg_freq)
    return series


class DataSet:
    def __init__(
        self, rid: int, series: list[str], pct_chg: Union[bool, list[bool]] = False
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
        data = dict()
        if hasattr(self.pct_chg, "__iter__"):
            for i, series in enumerate(self.series):
                data[series] = get_series(series, pct_chg=self.pct_chg[i])
            return pd.DataFrame(data)
        for series in self.series:
            data[series] = get_series(series, pct_chg=self.pct_chg)
        return pd.DataFrame(data)

    def get_release_dates(self):
        release_dates = pf.get_release_dates(release_id=self.rid, api_key=api_key)
        release_df = pd.DataFrame(release_dates["release_dates"]).set_index("date")
        return release_df

    def get_release_offsets(self):
        self.data["release_date"] = pd.NaT
        for date in self.data.index:
            pred = self.release_dates.index[
                self.release_dates.index.get_indexer([date], method="bfill")[0]
            ]
            self.data.loc[date, "release_date"] = pred
        self.data.set_index("release_date", inplace=True)


class DataCollection:
    def __init__(self, datasets: list[callable]):
        self.datasets = [dataset().get() for dataset in datasets]
        self.data = self.merge_datasets()
        self.min_date = self.data.index.min()
        self.max_date = self.data.index.max()
        self.daterange = pd.date_range(start=self.min_date, end=self.max_date, freq="D")
        self.data = self.data.groupby(level=0).first()
        self.data = self.data.reindex(self.daterange, fill_value=np.nan)
        self.data = self.data.ffill()
        self.data = self.data.dropna()

    def merge_datasets(self):
        merged = self.datasets[0].data
        for dataset in self.datasets[1:]:
            merged = pd.merge(
                merged, dataset.data, left_index=True, right_index=True, how="outer"
            )
        return merged


class DefinedDataSet:
    rid = 0
    date_file_offset = 0
    series = []
    pct_chg = False

    @classmethod
    def get(cls):
        return DataSet(cls.rid, cls.date_file_offset, cls.series, cls.pct_chg)


class CPIData(DefinedDataSet):
    rid = 10
    date_file_offset = 40
    series = ["CPILFESL", "CPIENGSL"]
    pct_chg = True


class FOMCData(DefinedDataSet):
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
    rid = 229
    data_file_offset = 40
    series = ["TTLCONS"]
    pct_chg = True


class JOLTSData(DefinedDataSet):
    rid = 192
    date_file_offset = 41
    series = ["JTSJOR", "JTSLDR", "JTSQUR", "JTSHIR"]
    pct_chg = False


class GDPData(DefinedDataSet):
    rid = 53
    date_file_offset = 60
    series = ["GDP"]
    pct_chg = True


class EMPLData(DefinedDataSet):
    rid = 50
    date_file_offset = 40
    series = ["UNRATE", "CIVPART", "CES0500000003", "AWHAETP", "PAYEMS"]
    pct_chg = [False, False, True, True, True]


class MANUData(DefinedDataSet):
    rid = 25
    date_file_offset = 56
    series = ["ISRATIO"]
    pct_chg = True


if __name__ == "__main__":
    datasets = [CPIData, CONSTRData, JOLTSData, GDPData, EMPLData, MANUData]
    data = DataCollection(datasets)

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
for series in market_data:
    market_df = market_df.merge(series, how="outer", left_index=True, right_index=True)
print(market_df)

master_df = pd.merge(
    data.data, market_df, how="outer", left_index=True, right_index=True
).ffill()
master_df.to_csv("data/macro.csv")
