from pathlib import Path
from itertools import product as prod
import time
import math
import re
import pytest

import numpy as np
import pandas as pd
import scipy.integrate as sci_integrate

from hydrodiy.io import iutils

from pyrezeq import model

from test_approx import generate_samples, reservoir_function

from pygme.models import gr4j

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
LOGGER = iutils.get_logger("integrate", flog=FTEST / "test_integrate.log")


def test_gr4jprod(allclose):
    X1s = np.linspace(10, 1000, 100)
    mod = gr4j.GR4J()
    nsubdiv = 1 # gr4jprod == original gr4j model

    for siteid in data_reader.SITEIDS:
        df = data_reader.get_data(siteid, "daily")
        df = df.loc["2022-02-25": "2022-03-01"]
        inputs = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
        mod.allocate(inputs, noutputs=mod.noutputsmax)

        cc = ["S", "PR", "AE", "PERC"]
        for X1 in X1s:
            # Run original GR4J
            mod.X1 = X1
            mod.initialise_fromdata()
            s0 = mod.states.S
            mod.run()
            expected = mod.to_dataframe().loc[:, cc]

            # Run alternative GR production with 1 subdiv
            gp = pd.DataFrame(model.gr4jprod(nsubdiv, X1, s0, inputs), \
                                                                columns=cc)
            assert allclose(gp, expected)
