import cvxpy as cp
import numpy as np
import csv

def load_mel_data(file_path: str) -> tuple:
    """
    Load Melbourne data from a CSV file into two numpy arrays.

    The CSV file should have two columns, and the function will skip the header row.
    The values in the first column will be loaded into the first array and the values
    in the second column will be loaded into the second array. Both columns should
    contain numerical data.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple of two numpy arrays. The first array contains the data from
        the first column of the CSV file, and the second array contains the data
        from the second column of the CSV file.
    """
    # Create empty lists for x and y
    x = []
    y = []

    # Open the CSV file and read it into the lists
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    # Convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Return the arrays
    return x, y
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




# Specify the path to your CSV file
file_path = "/Users/diegolara/PycharmProjects/using_gtreg/data/melbourne_data.csv"

x, y = load_mel_data(file_path)


TYX = np.array(...)  # replace with your array
tYX = np.array(...)  # replace with your array
result, e_value, h_value = solve_cvxpy(TYX, tYX, algorithm='ECOS')
