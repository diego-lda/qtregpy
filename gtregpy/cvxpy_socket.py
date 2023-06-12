import cvxpy as cp
import numpy as np

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
    grad = np.dot(TYX.T, e) + np.dot(tYX.T, h)
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

    return result, e.value, h.value



TYX = np.array(...)  # replace with your array
tYX = np.array(...)  # replace with your array
result, e_value, h_value = solve_cvxpy(TYX, tYX, algorithm='ECOS')
