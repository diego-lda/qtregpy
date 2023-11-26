import numpy as np
import cvxpy as cp
import math
import pandas as pd

from patsy import dmatrix
from scipy.interpolate import splev, splrep, BSpline, interp1d
from scipy.integrate import simps

from typing import Optional, Union
from pandas.core.frame import DataFrame

def calc_loglike(b: np.ndarray, tyx: np.ndarray, tYX: np.ndarray) -> float:
    """
    Calculates the log-likelihood function.

    Args:
        b (np.ndarray): Coefficient matrix.
        tyx (np.ndarray): TYX matrix.
        tYX (np.ndarray): tYX matrix.

    Returns:
        float: The resulting sum of the log-likelihood function.
    """
    k_gauss = math.log(1 / math.sqrt(2 * math.pi))
    e = np.matmul(tyx, b)
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

def calc_score(e: np.ndarray, eta: np.ndarray, tyx: np.ndarray) -> np.ndarray:
    """
    Calculate the score vector given error and eta values.

    Args:
        e (np.ndarray): The error vector.
        eta (np.ndarray): The eta vector.
        tyx (np.ndarray): The TYX matrix.

    Returns:
        np.ndarray: The calculated score vector.
    """
    grad = tyx.T @ e + tyx.T @ eta
    return grad

def get_dimensions(tyx: np.ndarray) -> tuple[int, int]:
    """
    Get the dimensions of the TYX matrix.

    Args:
        tyx (np.ndarray): The matrix for which dimensions need to be determined.

    Returns:
        tuple[int, int]: A tuple containing the number of rows and columns in the TYX matrix.
    """
    bdim = tyx.shape[1]
    nobs = tyx.shape[0]
    print("Problem dimensions are:", nobs, bdim)
    return nobs, bdim

def set_lamx(
    k_score: np.ndarray, 
    gam: Union[float, int, None] = None, 
    lam: Optional[np.ndarray] = None, 
    lam_vec: Optional[np.ndarray] = None, 
    nXs: Optional[int] = None, 
    nYS: Optional[int] = None, 
    zeros: Optional[np.ndarray] = None, 
    weights: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    Set the values of lamx based on Kscore, gam, lam, lam_vec, nXs, nYS, zeros, and weights.

    Args:
        k_score (np.ndarray): The array of k_scores.
        gam (Union[float, int, None], optional): The gam value. Defaults to None.
        lam (Optional[np.ndarray], optional): The lam matrix. Defaults to None.
        lam_vec (Optional[np.ndarray], optional): The lam_vec vector. Defaults to None.
        nXs (Optional[int], optional): The number of rows in lam. Defaults to None.
        nYS (Optional[int], optional): The number of columns in lam. Defaults to None.
        zeros (Optional[np.ndarray], optional): The zeros value. Defaults to None.
        weights (Optional[np.ndarray], optional): The weights vector. Defaults to None.

    Returns:
        Optional[np.ndarray]: The lamx vector, or None if conditions are not met.
    """
    lamx = None

    # Assuming k_score is a scalar for comparison
    if gam is not None and k_score > 0:
        lamx = np.asarray(k_score)
    elif gam is not None and k_score == 0 and gam > 0:
        lamspecified = isinstance(lam, np.ndarray) and lam_vec is None

        if not lamspecified:
            if not isinstance(lam, np.ndarray) and nXs is not None and nYS is not None:
                lam = np.zeros((nXs, nYS))

            if isinstance(lam, np.ndarray) and lam_vec is not None:
                # Assuming lam_vec has sufficient length and lam has the right shape
                for i in range(len(lam)):
                    lam[i] = lam_vec[5]  # gen
                lam[0, :] = lam_vec[4]  # row1
                lam[:, 0] = lam_vec[2]  # col1
                lam[:, 1] = lam_vec[3]  # col1
                lam[0, 0] = lam_vec[0]  # int
                lam[0, 1] = lam_vec[1]  # int

        if zeros is not None and len(zeros) > 0 and weights is not None:
            lamx = np.asarray(lam)[-zeros] * weights
        elif zeros is not None and len(zeros) == 0 and weights is not None:
            lamx = np.asarray(lam) * weights

    return lamx

def set_kscore(k_score: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Set the kth score of an array to zero and determine if score bounds exist.
    Args:
        k_score (np.ndarray): The array of k_scores.
    Returns:
        tuple[np.ndarray, bool]: A tuple containing the modified Kscore array and a boolean value indicating if score bounds exist.
    """
    if len(k_score) > 1:
        k_score = np.matrix(k_score).reshape(-1, 1)

    scorebounds = np.max(k_score) > 0

    return k_score, scorebounds

def get_xminmax(tyx: np.ndarray) -> tuple[float, float]:
    """
    Calculate the minimum and maximum values of the second column in TYX.
    Args:
        tyx (np.ndarray): The TYX matrix.
    Returns:
        tuple[float, float]: A tuple containing the minimum and maximum values.
    """
    xmin = np.min(tyx[:, 1])
    xmax = np.max(tyx[:, 1])

    return xmin, xmax

def get_dedygridp2(vec_b, sYgrid, matM, nYS, nXs, tyx):
    """
    Calculate the minimum value of BetaY using vec_b, sYgrid, matM, nYS, nXs, xmin, and xmax.
    Args:
        vec_b: The b vector.
        sYgrid: The sYgrid matrix.
        matM: The M matrix.
        nYS: The number of columns in sYgrid.
        nXs: The number of rows in sYgrid.
        tyx (np.ndarray): The TYX matrix.
    Returns:
        float: The minimum value of x1 or x2.
    """
    xmin, xmax = get_xminmax(tyx)
    betaY = sYgrid @ np.transpose(np.reshape(matM @ vec_b, (nYS, nXs)))
    x1 = betaY[:, 0] + betaY[:, 1] * xmin
    x2 = betaY[:, 0] + betaY[:, 1] * xmax

    return np.min([np.min(x1), np.min(x2)])

def get_dedygridpp(vec_b, vec_Xs, matM, nXs, nYS, sYgrid):
    """
    Calculate the minimum value of x using vec_b, vec_Xs, matM, nXs, nYS, and sYgrid.
    Args:
        vec_b: The b vector.
        vec_Xs: The Xs list or array.
        matM: The M matrix.
        nXs: The number of rows in Xs.
        nYS: The number of columns in Xs.
        sYgrid: The sYgrid matrix.
    Returns:
        float: The minimum value of x.
    """
    now_Xs = None
    if isinstance(vec_Xs, list):
        for kk in range(len(vec_Xs)):
            if isinstance(vec_Xs[kk], np.ndarray):
                for jj in range(vec_Xs[kk].shape[2]):
                    now_Xs = np.vstack((now_Xs, vec_Xs[kk][:, :, jj])) if now_Xs is not None else vec_Xs[kk][:, :, jj]
            if not isinstance(vec_Xs[kk], np.ndarray):
                now_Xs = np.vstack((now_Xs, vec_Xs[kk])) if now_Xs is not None else vec_Xs[kk]
    else:
        now_Xs = vec_Xs

    if not isinstance(vec_Xs, list):
        now_Xs = vec_Xs

    beta = now_Xs @ np.reshape(matM @ vec_b, (nXs * nYS, 1))

    x = beta @ np.transpose(sYgrid)
    return np.min(x)

def get_dedygrid21(vec_b, sYgrid, matM, nYS, nXs):
    """
    Calculate the minimum value of BetaY using vec_b, sYgrid, matM, nYS, and nXs.
    Args:
        vec_b: The b vector.
        sYgrid: The sYgrid matrix.
        matM: The M matrix.
        nYS: The number of columns in sYgrid.
        nXs: The number of rows in sYgrid.
    Returns:
        float: The minimum value of BetaY.
    """
    BetaY = sYgrid @ np.transpose(np.reshape(matM @ vec_b, (nYS, nXs)))
    return np.min(BetaY)

def get_beta_2x(vec_b, vec_Xs, M, nXs, nYS):
    """
    Calculate the minimum value of Beta using vec_b, Xs, M, nXs, and nYS.
    Args:
        vec_b: The b vector.
        vec_Xs: The Xs list or array.
        M: The M matrix.
        nXs: The number of rows in Xs.
        nYS: The number of columns in Xs.
    Returns:
        float: The minimum value of Beta.
    """
    now_Xs = None
    if isinstance(vec_Xs, list):
        for kk in range(len(vec_Xs)):
            if isinstance(vec_Xs[kk], np.ndarray):
                for jj in range(vec_Xs[kk].shape[2]):
                    now_Xs = np.vstack((now_Xs, vec_Xs[kk][:, :, jj])) if now_Xs is not None else vec_Xs[kk][:, :, jj]
            if not isinstance(vec_Xs[kk], np.ndarray):
                now_Xs = np.vstack((now_Xs, vec_Xs[kk])) if now_Xs is not None else vec_Xs[kk]
    else:
        now_Xs = vec_Xs

    Mb = M @ vec_b
    Mb_reshaped = cp.reshape(Mb, (nXs, nYS))

    Beta = now_Xs @ Mb_reshaped

    return np.min(Beta[:, 1])

def set_constraints(pen, bounded, beta2, vec_b, c_bound, cval, vec_Xs, matM, nXs, nYS, sYgrid, tyx):
    """
    Set the constraints based on the given conditions.
    Args:
        pen: The pen value.
        bounded: Boolean indicating if bounded constraints are applied.
        beta2: Boolean indicating if beta2 constraints are applied.
        vec_b: The b vector.
        c_bound: The c_bound value.
        cval: The cval value.
        vec_Xs: The Xs value.
        matM: The matrix M.
        nXs: the nXs value.
        nYS: the nYS value.
        sYgrid: the sYgrid matrix.
        tyx: The TYX matrix.
    Returns:
        list: A list of constraint conditions.
    """
    constr = []

    if len(pen) == 0 and bounded:
        constraint_condns = np.linalg.norm(vec_b) <= c_bound
        constr.append(constraint_condns)

    if len(pen) == 0 and beta2:
        constraint_condns = get_beta_2x(vec_b) >= cval
        constr.append(constraint_condns)

    if len(pen) > 0 and not bounded and not beta2:
        if nXs == 1  and nYS == 2:
            constraint_condns = get_dedygrid21(vec_b, sYgrid, matM, nYS, nXs) >= cval
        elif nXs > 1 and nYS == 2:
            constraint_condns = get_beta_2x(vec_b, vec_Xs, matM, nXs, nYS) >= cval
        elif nXs > 2 and nYS > 2:
            constraint_condns = get_dedygridpp(vec_b, vec_Xs, matM, nXs, nYS, sYgrid) >= cval
        elif nXs == 2 and nYS > 2:
            constraint_condns = get_dedygridp2(vec_b, sYgrid, matM, nYS, nXs, tyx) >= cval
        else:
            constraint_condns = get_beta_2x(vec_b, vec_Xs, matM, nXs, nYS) >= cval
        constr.append(constraint_condns)

    if len(pen) > 0 and bounded and not beta2:
        if nXs == 1  and nYS == 2:
            constraint_condns1 = get_dedygrid21(vec_b, sYgrid, matM, nYS, nXs) >= cval
        elif nXs > 1 and nYS == 2:
            constraint_condns1 = get_beta_2x(vec_b, vec_Xs, matM, nXs, nYS) >= cval
        elif nXs > 2 and nYS > 2:
            constraint_condns1 = get_dedygridpp(vec_b, vec_Xs, matM, nXs, nYS, sYgrid) >= cval
        elif nXs == 2 and nYS > 2:
            constraint_condns1 = get_dedygridp2(vec_b, sYgrid, matM, nYS, nXs, tyx) >= cval
        else:
            constraint_condns1 = get_beta_2x(vec_b, vec_Xs, matM, nXs, nYS) >= cval
        constraint_condns2 = np.linalg.norm(vec_b) <= c_bound
        constr.extend([constraint_condns1, constraint_condns2])

    if len(pen) > 0 and not bounded and beta2:
        if nXs == 1 and nYS == 2:
            constraint_condns1 = get_dedygrid21(vec_b, sYgrid, matM, nYS, nXs) >= cval
        elif nXs > 1 and nYS == 2:
            constraint_condns1 = get_beta_2x(vec_b, vec_Xs, matM, nXs, nYS) >= cval
        elif nXs > 2 and nYS > 2:
            constraint_condns1 = get_dedygridpp(vec_b, vec_Xs, matM, nXs, nYS, sYgrid) >= cval
        elif nXs == 2 and nYS > 2:
            constraint_condns1 = get_dedygridp2(vec_b, sYgrid, matM, nYS, nXs, tyx) >= cval
        else:
            constraint_condns1 = get_beta_2x(vec_b, vec_Xs, matM, nXs, nYS) >= cval
        constraint_condns2 = get_beta_2x(vec_b, vec_Xs, matM, nXs, nYS) >= cval
        constr.extend([constraint_condns1, constraint_condns2])

    return constr

def generate_xs(x: np.ndarray = None) -> DataFrame:
    """
    Generates a design matrix from a given NumPy array.

    This function converts a NumPy array into a pandas DataFrame and then uses patsy's 
    dmatrix function to create a design matrix with an intercept and the x variable.

    Args:
        x (np.ndarray, optional): A NumPy array. Defaults to None.

    Returns:
        DataFrame: A pandas DataFrame representing the design matrix.

    Examples:
        >>> import numpy as np
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> xs = generate_xs(x)
        >>> print(xs)
        
        # The output will be a DataFrame with an intercept column (of 1s) and a column for `x`.
    """
    if x is None:
        raise ValueError("Input array 'x' cannot be None")

    # Converting the numpy array to a pandas DataFrame
    x_df = pd.DataFrame({'x': x})
    
    # Creating a design matrix using patsy
    xs = dmatrix('~ x', data=x_df, return_type='dataframe')
    
    return xs

def tz_form(vec_a: Union[np.ndarray, list], vec_b: Union[np.ndarray, list]) -> np.ndarray:
    """
    Generates the TZ matrix by multiplying each column in X with each column in Y.

    This function takes two arrays (or lists that can be converted to arrays), X and Y. If X and Y are not
    matrices, they are converted into matrices. The function then generates a matrix TZ where each column
    is the product of each column in X with each column in Y.

    Args:
        vec_a (Union[np.ndarray, list]): Input matrix X or a list that can be converted to a matrix.
        vec_b (Union[np.ndarray, list]): Input matrix Y or a list that can be converted to a matrix.

    Returns:
        np.ndarray: The resulting TZ matrix.

    Examples:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> Y = np.array([[5, 6], [7, 8]])
        >>> tz_form(X, Y)
        # Output: array([[ 5,  6, 10, 12],
        #                [21, 24, 28, 32]])
    """
    # Convert X and Y to matrices if they are not already
    vec_b = np.atleast_2d(vec_b)
    if not isinstance(vec_a, np.ndarray):
        vec_a = np.array(vec_a)
    if vec_a.ndim == 1 or not isinstance(vec_a, np.ndarray):
        vec_a = np.tile(vec_a, (vec_b.shape[0], 1))

    na = vec_a.shape[1]
    nb = vec_b.shape[1]
    nobs = vec_a.shape[0]
    mat_tz = np.zeros((nobs, na * nb))

    i = 0
    for j in range(nb):
        for k in range(na):
            mat_tz[:, i] = vec_a[:, k] * vec_b[:, j]
            i += 1

    return mat_tz

def round_any(value: float, base: float, method) -> float:
    """
    Round a value to any base using a specified method (floor or ceiling).

    Args:
        value (float): The value to round.
        base (float): The base to round to.
        method: The method to use for rounding, e.g., np.floor or np.ceil.

    Returns:
        float: The rounded value.

    Example:
        >>> round_any(3.123, 0.001, np.floor)
        3.123
    """
    return method(value / base) * base

def generate_y_splines(y: np.ndarray, ydf: int, yorder: int) -> tuple:
    """
    Translate R spline basis generation code to Python using numpy and scipy.

    Args:
        y (np.ndarray): Array of data points.
        ydf (int): Degrees of freedom for the spline.
        yorder (int): Order of the spline.

    Returns:
        tuple: A tuple containing various spline related matrices and values.

    Example:
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> generate_y_splines(y, 4, 4)
        (array([[...]]), array([[...]]), ...)
    """
    iyknots = np.percentile(y, np.linspace(0, 100, ydf))
    iyknots[0] = round_any(iyknots[0], 0.001, np.floor)
    iyknots[-1] = round_any(iyknots[-1], 0.001, np.ceil)
    iyknots = np.round(iyknots, 3)
    
    # Expand and evaluate knots
    # Note: Python's BSpline does not have an equivalent for `expand.knots` directly.
    # We'll use a BSpline constructor for this purpose.
    t = np.linspace(iyknots[0], iyknots[-1], ydf)
    c = np.zeros(ydf)
    ys = BSpline(t, c, yorder)

    # Integrate and evaluate spline
    yls = ys.integrate(0, 1)  # Adjust the limits as per your requirement
    lys = splev(y, ys)
    ys1 = splev(y, yls)

    lyls = ys1
    lyls[:, 0] = y
    lyls = np.c_[np.ones(len(y)), lyls]
    lys = lys[:, 1:]
    lys = np.c_[np.ones(len(lys)), lys]
    nys = lyls.shape[1]

    nobs = len(y)
    ygrid = y  # Assuming ygrid is equivalent to y in this context
    syfunc = yls
    yfunc = ys
    lslyy = splev(ygrid, syfunc)
    slyy = splev(ygrid, yfunc)

    lslygrid = np.c_[np.ones(len(ygrid)), ygrid, lslyy[:, 1:]]
    slygrid = np.c_[np.zeros(len(slyy)), np.ones(len(slyy)), slyy[:, 1:]]

    return lyls, lys, nys, nobs, ygrid, syfunc, yfunc, lslyy, slyy, lslygrid, slygrid

def create_diagonal_matrix(nxs: int, nys: int) -> np.ndarray:
    """
    Create a square diagonal matrix with ones on the diagonal and zeros elsewhere.

    Args:
        nxs (int): The first dimension integer.
        nys (int): The second dimension integer.

    Returns:
        np.ndarray: The resulting square diagonal matrix.

    Example:
        >>> create_diagonal_matrix(3, 2)
        array([[1., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 0., 1.]])
    """
    size = nxs * nys
    return np.eye(size)

def prepare_gtr_primal(x: np.ndarray, y: np.ndarray, tyx: np.ndarray, k_score: np.ndarray):
    xs = generate_xs(x)
    nxs = xs.shape[1]
    nobs, bdim = get_dimensions(tyx)
    k_score, scorebounds = set_kscore(k_score)
    k_gauss = math.log(1 / math.sqrt(2 * math.pi))
    vec_b = cp.Variable(bdim)
    # set_problemfns()
    lam = 0
    gam = 0
    lam_vec = 0
    
    ydf = 4
    yorder = 4
    lyls, lys, nys, nobs, ygrid, syfunc, yfunc, lslyy, slyy, lslygrid, slygrid = generate_y_splines(y, ydf, yorder)

    btarg = np.zeros(nxs * nys)
    egam = 0
    
    ltlz = tz_form(vec_a=xs,vec_a=lyls)
    tlz = tz_form(vec_a=xs,vec_a=lys)
    
    ltlylx = ltlz
    tlylx = tlz
    
    mat_m = create_diagonal_matrix(nxs, nys)
    
    lamx = set_lamx(k_score, gam, lam, lam_vec, nxs, nys)
    reg = gam * np.sum(lamx * np.abs(vec_b))
    elastic = egam * np.linalg.norm(vec_b - btarg, 2)
    
    obj = calc_loglike(vec_b, ltlylx, tlylx) - reg - elastic
    
    pen = 0
    beta2 = False
    bounded = False
    c_bound = 1e6
    cval = 0.1
    
    constraints = set_constraints(pen, bounded, beta2, vec_b, c_bound, cval, xs, mat_m, nxs, nys, slygrid, ltlylx)
