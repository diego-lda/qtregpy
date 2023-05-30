import numpy as np
import math

import cvxpy as cp
import scipy.sparse as sp

def print_name(name):
    """Prints the input name within a string.

    This function takes a string input 'name' and prints it
    within another string.

    Args:
        name: A string input.

    Returns:
        None
    """
    print(f"Hello, {name}!")
def calc_loglike(b: np.ndarray, TYX: np.ndarray, tYX: np.ndarray) -> float:
    """
    Calculates the log-likelihood function.

    Args:
        b (np.ndarray): Coefficient matrix.
        TYX (np.ndarray): TYX matrix.
        tYX (np.ndarray): tYX matrix.

    Returns:
        float: The resulting sum of the log-likelihood function.
    """
    Kgauss = math.log(1 / math.sqrt(2 * math.pi))
    e = np.matmul(TYX, b)
    dedy = np.matmul(tYX, b)
    llfvec = -.5 * e ** 2 + np.log(dedy) + Kgauss
    return np.sum(llfvec)