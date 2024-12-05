import importlib

__version__ = "1.2"

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


