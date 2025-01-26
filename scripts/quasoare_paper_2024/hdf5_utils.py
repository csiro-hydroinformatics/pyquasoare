import numpy as np
import pandas as pd
import tables

TIME_START = pd.to_datetime("1970-01-01")
NFLUX_MAX = 4

COMPRESSION_FILTER = tables.Filters(complevel=9)

class SimDescription(tables.IsDescription):
    time = tables.Int64Col(pos=0)
    flux1 = tables.Float64Col(pos=1)
    flux2 = tables.Float64Col(pos=2)
    flux3 = tables.Float64Col(pos=3)
    flux4 = tables.Float64Col(pos=4)
    s1 = tables.Float64Col(pos=5)
    niter = tables.Int32Col(pos=6)


SIM_DTYPE = np.dtype([("time", np.int64), \
                        ("flux1", np.float64), \
                        ("flux2", np.float64), \
                        ("flux3", np.float64), \
                        ("flux4", np.float64), \
                        ("s1", np.float64), \
                        ("niter", np.int32)])

class FluxesDescription(tables.IsDescription):
    s = tables.Float64Col(pos=0)
    flux1_true = tables.Float64Col(pos=1)
    flux1_approx = tables.Float64Col(pos=2)
    flux2_true = tables.Float64Col(pos=3)
    flux2_approx = tables.Float64Col(pos=4)
    flux3_true = tables.Float64Col(pos=5)
    flux3_approx = tables.Float64Col(pos=6)
    flux4_true = tables.Float64Col(pos=7)
    flux4_approx = tables.Float64Col(pos=8)


FLUXES_DTYPE = np.dtype([("s", np.float64), \
                        ("flux1_true", np.float64), \
                        ("flux1_approx", np.float64), \
                        ("flux2_true", np.float64), \
                        ("flux2_approx", np.float64), \
                        ("flux3_true", np.float64), \
                        ("flux3_approx", np.float64), \
                        ("flux4_true", np.float64), \
                        ("flux4_approx", np.float64)])


def format_sim(time_index, sim, s1, niter):
    nval, nfluxes = sim.shape
    simdata = np.empty(dtype=SIM_DTYPE, shape=nval)
    simdata["time"] = (time_index-TIME_START).total_seconds()\
                            .astype(np.int64)
    simdata["s1"] = s1
    simdata["niter"] = niter
    for ifx in range(NFLUX_MAX):
        if ifx<nfluxes:
            simdata[f"flux{ifx+1}"] = sim[:, ifx]
        else:
            simdata[f"flux{ifx+1}"] = np.nan

    return simdata


def store(h5, group, table_name, data, description):
    nval = len(data)
    tb = h5.create_table(group, table_name, \
                        description=description, \
                        expectedrows=nval, \
                        filters=COMPRESSION_FILTER)
    tb.append(data)
    return tb


def addmeta(tb, **kwargs):
    for nm, value in kwargs.items():
        tb.attrs[nm] = value


def convert(tb):
    tb = pd.DataFrame(tb)
    if hasattr(tb, "time"):
        tb["time"] = pd.to_datetime(tb.time.astype(np.float64), unit="s")
        tb = tb.set_index("time")
    return tb


def read(group, table_name):
    tb = group[table_name][:]
    return convert(tb)
