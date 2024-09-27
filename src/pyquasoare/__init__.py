import importlib

def has_c_module(raise_error=True):
    name = f"c_pyquasoare"
    out = importlib.util.find_spec(name)

    if not out is None:
        return True
    else:
        if raise_error:
            raise ImportError(f"C module c_pyquasoare is "+\
                "not available, please run python setup.py build")
        else:
            return False

#from ._version import get_versions
#__version__ = get_versions()['version']
#del get_versions

from . import _version
__version__ = _version.get_versions()['version']
