from pathlib import Path
from itertools import product as prod
import importlib
import numpy as np
import pandas as pd

from hydrodiy.io import csv

FROOT = Path(__file__).resolve().parent.parent.parent
FREADER = FROOT / "tests" / "data_reader.py"
spec = importlib.util.spec_from_file_location("data_reader", FREADER)
data_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_reader)

SITEIDS = data_reader.SITEIDS

MODEL_NAMES = ["QR", "CR", "BCR", "GR", "GRM"]

ODE_METHODS = ["analytical", "radau", "rk45", "dop853"]

NALPHAS = [5, 10, 50, 500]
ODE_METHODS += [f"{v}_quasoare_{n}" for v in["py", "c"] for n in NALPHAS]
ODE_METHODS += [f"{v}_quasoarelin_{n}" for v in["py"] for n in NALPHAS]

CONFIGS = np.array([(m, s) for m, s in prod(MODEL_NAMES, SITEIDS)])


def get_config(config):
    if config>=CONFIGS.shape[0]:
        raise ValueError(f"Cannot find config {config}")
    return CONFIGS[config]


def get_data(siteid, timestep):
    return data_reader.get_data(siteid, timestep)


def get_sites():
    fs = FROOT / "data" / "sites.csv"
    sites, _ = csv.read_csv(fs, index_col="STATIONID", \
                    dtype={"STATIONID": str, "UPSTREAM_STATIONID":str})
    return sites.loc[SITEIDS]
