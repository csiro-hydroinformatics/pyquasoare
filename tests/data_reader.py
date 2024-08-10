from pathlib import Path

from hydrodiy.io import csv

source_file = Path(__file__).resolve()
FTEST = source_file.parent

SITEIDS = ["203004", "203005", "203014", "203024", \
                "203060", "203900"]
TIMESTEPS = ["daily", "hourly"]

def get_data(siteid, timestep):
    errmess = f"Cannot find siteid {siteid}"
    assert siteid in SITEIDS, errmess
    errmess = f"Cannot find timestep {timestep}"
    assert timestep in TIMESTEPS

    fd = FTEST.parent / "data" / timestep / f"hydrodata_{siteid}.csv"
    df, _ = csv.read_csv(fd, index_col=0, parse_dates=True)
    return df
