from tqdm import tqdm, tqdm_notebook
from .constants import VerbosityLevel


def numerical_order(files):
    return sorted(files, key=lambda x: int(x.split('.')[0]))

def in_jupyter():
    try:
        get_ipython()
        return True
    except Exception:
        return False


def make_verbose():
    if in_jupyter():
        return VerbosityLevel.JUPYTER
    else:
        return VerbosityLevel.CONSOLE


def wrap_with_tqdm(it, verbosity=make_verbose(), **kwargs):
    if verbosity == VerbosityLevel.SILENT:
        return it
    elif verbosity == VerbosityLevel.JUPYTER:
        return tqdm_notebook(it, **kwargs)
    elif verbosity == VerbosityLevel.CONSOLE:
        return tqdm(it, **kwargs)
