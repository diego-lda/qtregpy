import cvxpy as cp
import numpy as np
import csv
from scipy.interpolate import BSpline
from scipy.stats import norm

def score(e: np.array, h: np.array, TYX: np.array, tYX: np.array) -> np.array:
    """
    Compute the product of two matrices.

    Args:
        e (np.array): First input vector.
        h (np.array): Second input vector.
        TYX (np.array): First input matrix.
        tYX (np.array): Second input matrix.

    Returns:
        np.array: The product of two matrices.
    """
    grad = TYX.T @ e + tYX.T @ h
    return grad

def get_dllf(e: cp.Variable, h: cp.Variable, Kgauss: float = np.log(1/np.sqrt(2*np.pi))) -> cp.Expression:
    """
    Compute the sum of a vector.

    Args:
        e (cp.Variable): First input variable.
        h (cp.Variable): Second input variable.
        Kgauss (float, optional): Constant for the calculation. Defaults to np.log(1/np.sqrt(2*np.pi)).

    Returns:
        cp.Expression: The computed sum.
    """
    llfvec = 0.5 * cp.square(e) - cp.log(-h) - 1 + Kgauss
    return cp.sum(llfvec)

def solve_cvxpy(TYX: np.ndarray, tYX: np.ndarray, algorithm: str = None) -> tuple:
    """
    Solve a convex optimization problem.

    Args:
        TYX (np.ndarray): First array input.
        tYX (np.ndarray): Second array input.
        algorithm (str, optional): The solver algorithm to use. Defaults to None.

    Returns:
        tuple: The result of the problem, the value of e and the value of h.
    """
    nobs = TYX.shape[0]
    e = cp.Variable(nobs)
    h = cp.Variable(nobs)

    cond1 = score(e, h, TYX, tYX) == 0
    constraints = [cond1]
    obj = cp.Minimize(get_dllf(e, h))
    problem = cp.Problem(obj, constraints)

    solver_map = {
        'ECOS': cp.ECOS,
        'OSQP': cp.OSQP,
        'SCS': cp.SCS
    }

    if algorithm is not None:
        result = problem.solve(solver=solver_map.get(algorithm))
    else:
        result = problem.solve()

    # analyse the results
    shadow1 = cond1.dual_value
    bhat = -shadow1

    return result, e.value, h.value, bhat

def form_tz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Generate the tz matrix.

    If x or y are not 2D arrays (i.e., matrices), they are reshaped to be 2D.
    The tz matrix is formed by multiplying each column of x with each column of y.

    Args:
        x (np.ndarray): First input array.
        y (np.ndarray): Second input array.

    Returns:
        np.ndarray: The tz matrix.
    """
    # Ensure X and Y are 2D arrays
    if len(y.shape) == 1:
        y = y.reshape((1, -1))
    if len(x.shape) == 1:
        x = x.reshape((y.shape[0], -1))

    nx = x.shape[1]
    ny = y.shape[1]
    nobs = x.shape[0]

    tz = np.zeros((nobs, nx * ny))
    i = 0

    for j in range(ny):
        for k in range(nx):
            tz[:, i] = x[:, k] * y[:, j]
            i += 1

    return tz

def get_iyknots(y: np.ndarray, y_order: int) -> np.ndarray:
    """
    Calculate quantiles for the given array.

    Args:
        y (np.ndarray): Input array for which to calculate quantiles.
        y_order (int): The number of quantiles to calculate.

    Returns:
        np.ndarray: Array of quantile values.
    """
    return np.quantile(y, np.linspace(0, 1, y_order))

def create_spline_basis(knots: np.ndarray, order: int):
    """
    Create B-spline basis functions for given knots and order.

    Args:
        knots (np.ndarray): Input array of knot values.
        order (int): Order of the spline, or spline degree + 1.

    Returns:
        list: List of B-spline basis functions.
    """
    nknots = len(knots)
    bsplines = [BSpline.basis_element(knots[i:i+order+1]) for i in range(nknots - order)]
    return bsplines

def _process_inputs(x, y, big_tyx, low_tyx):
    # Check if c and d are not None
    if big_tyx is not None and low_tyx is not None:
        return big_tyx, low_tyx

    # If a and b are None, print an error message
    if x is None or y is None:
        print("Error: Didn't pass the right data.")
        return None

    # Build the xs matrix
    xs = np.concatenate((np.ones((x.shape[0], 1)), x.reshape(-1, 1)), axis=1)

    # Build Ys and ys matrices
    nobs = len(y)
    cap_ys = np.c_[np.ones(len(y)), y]
    low_ys = np.array([0, 1] * nobs).reshape(nobs, 2)

    # Build the TYX and tYX matrices
    big_tyx = form_tz(x=xs, y=cap_ys)
    low_tyx = form_tz(x=xs, y=low_ys)

    return big_tyx, low_tyx

def compute_basic(x: np.ndarray = None, y: np.ndarray = None, big_tyx: np.ndarray = None, low_tyx: np.array = None) -> dict:
    """
    Perform a series of operations and return the results in a dictionary.

    Args:
        x (np.ndarray): Input array.
        y (np.ndarray): Input array.
        big_tyx (np.ndarray): The TYX matrix.
        low_tyx (np.ndarray): The tYX matrix.

    Returns:
        dict: A dictionary containing various computed values.
    """
    
    b_tyx, l_tyx = _process_inputs(x, y, big_tyx, low_tyx)
    
    # Solve the problem
    result, e_value, h_value, b_hat = solve_cvxpy(b_tyx, l_tyx, algorithm='ECOS')

    # Manipulate results to return
    finalscore = score(e_value, h_value, b_tyx, l_tyx)
    objval = get_dllf(e_value, h_value)
    llf_vec = np.log(norm.pdf(e_value) * (-1/h_value))
    llf = np.sum(llf_vec)
    e_hat = -np.dot(b_tyx, b_hat)
    eta_hat = np.dot(l_tyx, b_hat)

    ans = {
        "llf": llf,
        "e": e_value,
        "eta": -1/h_value,
        "finalscore": finalscore,
        "result": result,
        "e_hat": e_hat,
        "eta_hat": eta_hat,
        "h": h_value,
        "objval": objval,
        "llf_vec": llf_vec,
        "b_mat": b_hat
    }

    return ans