from pathlib import Path

source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

import importlib.util
spec = importlib.util.spec_from_file_location("data_utils",
                                              froot / "scripts" / "data_utils.py")
data_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_utils)

spec = importlib.util.spec_from_file_location("hdf5_utils",
                                              froot / "scripts" / "hdf5_utils.py")
hdf5_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hdf5_utils)

