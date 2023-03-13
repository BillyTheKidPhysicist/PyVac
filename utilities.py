import numpy as np

def is_ascending(vals) -> bool:
    """Return True if vals is in ascending order, else False"""
    if isinstance(vals, np.ndarray):
        return np.all(np.sort(vals) == vals)
    else:
        return all(val1 == val2 for val1, val2 in zip(vals, sorted(vals)))