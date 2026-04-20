from pathlib import Path
import math
import re
import pytest
import subprocess

import numpy as np
import pandas as pd

from hydrodiy.io import iutils

from hydrodiy.io import csv

source_file = Path(__file__).resolve()
FTEST = source_file.parent

def test_readme_snipet():
    freadme = FTEST.parent / "README.md"
    with freadme.open("r") as fo:
        txt = fo.readlines()

    istarts = [iline + 1 for iline, line in enumerate(txt)
              if line.strip() == "```python"]

    # Test each snipet
    for isnip, istart in enumerate(istarts):
        # Find end line
        iend = istart + next(iline for iline, line in enumerate(txt[istart:])
                             if line.strip() == "```")

        # Write code
        fsn = FTEST / "snipets" / f"README_{isnip}.py"
        fsn.parent.mkdir(exist_ok=True)
        with fsn.open("w") as fo:
            fo.write("".join(txt[istart: iend]))

        # Style
        cmd = f"flake8 {fsn}"
        subprocess.check_call(cmd, shell=True)

        # Run
        cmd = f"python {fsn}"
        subprocess.check_call(cmd, shell=True)

