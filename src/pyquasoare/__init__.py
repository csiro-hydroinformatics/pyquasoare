import importlib

__version__ = "1.2"


def has_c_module(raise_error=True):
    mname = "c_pyquasoare"
    out = importlib.util.find_spec(mname)

    if out is not None:
        return True
    else:
        if raise_error:
            raise ImportError(f"C module {mname} is not available."
                              + " Please run python setup.py build")
        else:
            return False
