from pathlib import Path

import pandas as pd

from hydrodiy.io import csv

source_file = Path(__file__).resolve()
FTEST = source_file.parent

TIMESTEPS = ["daily", "hourly"]


def get_stations():
    fs = FTEST.parent / "data" / "sites.csv"
    stations, _ = csv.read_csv(fs, dtype={"STATIONID": str},
                               index_col="STATIONID")
    return stations.iloc[:, :6]


SITEIDS = get_stations().index.to_list()


def get_data(siteid, timestep, nval=365 * 5):
    errmess = f"Cannot find siteid {siteid}"
    assert siteid in SITEIDS, errmess
    errmess = f"Cannot find timestep {timestep}"
    assert timestep in TIMESTEPS

    fd = FTEST.parent / "data" / timestep / f"hydrodata_{siteid}.csv"
    df, _ = csv.read_csv(fd, index_col=0, parse_dates=True)
    if nval is None:
        return df
    else:
        nval = nval * 24 if timestep == "hourly" else nval
        return df.iloc[-nval:]
