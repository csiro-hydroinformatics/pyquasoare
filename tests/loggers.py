from pathlib import Path
import re

from hydrodiy.io import iutils

FTEST = Path(__file__).resolve().parent

LOGGERS = {}

def get_logger(debug_mode=False, logger_name="default"):
    if debug_mode:
        key = f"{logger_name}-debug"
        console = True
        flog = None
    else:
        key = f"{logger_name}-normal"
        console = False
        flog = FTEST / f"test_{logger_name}.log"

    if key not in LOGGERS:
        logger = iutils.get_logger(logger_name,
                                   console=console,
                                   flog=flog)
        LOGGERS[key] = logger

    return LOGGERS[key]

