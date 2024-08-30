from pathlib import Path
from itertools import product as prod
import importlib
import numpy as np
import pandas as pd

FROOT = Path(__file__).resolve().parent.parent
FREADER = FROOT / "tests" / "data_reader.py"
spec = importlib.util.spec_from_file_location("data_reader", FREADER)
data_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_reader)

SITEIDS = data_reader.SITEIDS

MODEL_NAMES = ["QR", "BCR", "GRP", "GRPM"]

ODE_METHODS = ["analytical", "radau", "rk45", \
                "py_quasoare_5", "py_quasoare_100", \
                "c_quasoare_5", "c_quasoare_100"]

CONFIGS = np.array([(m, s) for m, s in prod(MODEL_NAMES, SITEIDS)])


def get_config(config):
    if config>=CONFIGS.shape[0]:
        raise ValueError(f"Cannot find config {config}")
    return CONFIGS[config]


def get_data(siteid, timestep):
    return data_reader.get_data(siteid, timestep)
