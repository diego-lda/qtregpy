import numpy as np
import cvxpy as cp
import math

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
    k_gauss = math.log(1 / math.sqrt(2 * math.pi))
    e = np.matmul(TYX, b)
    dedy = calc_dedy(tYX, b)
    llfvec = -.5 * e ** 2 + np.log(dedy) + k_gauss
    return np.sum(llfvec)

def calc_dedy(tYX: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate dedy vector given b values.

    Args:
        b (np.ndarray): The b vector.
        tYX (np.ndarray): The tYX matrix.

    Returns:
        np.ndarray: The calculated dedy vector.
    """
    dedy = np.matmul(tYX, b)
    return dedy

def calc_score(e: np.ndarray, eta: np.ndarray, TYX: np.ndarray) -> np.ndarray:
    """
    Calculate the score vector given error and eta values.

    Args:
        e (np.ndarray): The error vector.
        eta (np.ndarray): The eta vector.
        TYX (np.ndarray): The TYX matrix.

    Returns:
        np.ndarray: The calculated score vector.
    """
    grad = TYX.T @ e + TYX.T @ eta
    return grad

def get_dimensions(TYX: np.ndarray) -> tuple[int, int]:
    """
    Get the dimensions of the TYX matrix.

    Args:
        TYX (np.ndarray): The matrix for which dimensions need to be determined.

    Returns:
        tuple[int, int]: A tuple containing the number of rows and columns in the TYX matrix.
    """
    bdim = TYX.shape[1]
    nobs = TYX.shape[0]
    print("Problem dimensions are:", nobs, bdim)
    return nobs, bdim
