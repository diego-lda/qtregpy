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

def set_lamx(k_score: np.ndarray, gam, lam, lam_vec, nXs, nYS, zeros, weights) -> np.ndarray:
    """
    Set the values of lamx based on Kscore, gam, lam, lam_vec, nXs, nYS, zeros, and weights.
    Args:
        k_score (np.ndarray): The array of k_scores.
        gam: The gam value.
        lam: The lam matrix.
        lam_vec: The lam_vec vector.
        nXs: The number of rows in lam.
        nYS: The number of columns in lam.
        zeros: The zeros value.
        weights: The weights vector.
    Returns:
        np.ndarray: The lamx vector.
    """
    lamx = None

    if k_score > 0:
        lamx = np.asarray(k_score)
    elif k_score == 0 and gam > 0:
        lamspecified = isinstance(lam, np.ndarray) and lam_vec is None

        if not lamspecified:
            if not isinstance(lam, np.ndarray):
                lam = np.matrix(lam, nr=nXs, nc=nYS)

            if isinstance(lam, np.ndarray):
                for i in range(len(lam)):
                    lam[i] = lam_vec[5]  # gen
                lam[0, :] = lam_vec[4]  # row1
                lam[:, 0] = lam_vec[2]  # col1
                lam[:, 1] = lam_vec[3]  # col1
                lam[0, 0] = lam_vec[0]  # int
                lam[0, 1] = lam_vec[1]  # int

        if len(zeros) > 0:
            lamx = np.asarray(lam)[-zeros] * weights
        elif len(zeros) == 0:
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
        constraint_condns = np.linalg.norm(b) <= c_bound
        constr.append(constraint_condns)

    if len(pen) == 0 and beta2:
        constraint_condns = get_beta_2x(b) >= cval
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
