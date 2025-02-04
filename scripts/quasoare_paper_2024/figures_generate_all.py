#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-01-23 Thu 04:41 PM
## Comment : Generate all figures
##
## ------------------------------

import re
from pathlib import Path
import subprocess

from hydrodiy.io import iutils

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

#------------------------------------------------------------
# @Logging
#------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#------------------------------------------------------------
# @Process
#------------------------------------------------------------
lf = source_file.parent.glob("*.py")
for f in lf:
    if not re.search("^(figure|supplementary)_", f.stem):
        continue

    LOGGER.info(f"\n\n%%%%%%%%%%%%%%% Running [{f.stem}] %%%%%%%%%%%%%%%\n")
    args = f"python {f}"
    if f.stem.startswith("figure_B"):
        args += " -m GR -s 203024 -i 4"

    proc = subprocess.Popen(args, shell=True)
    proc.communicate()

LOGGER.completed()
