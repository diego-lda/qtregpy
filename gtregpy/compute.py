import numpy as np
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
def gtr_compute(TYX, tYX, y, x, nyg=0, ng_qgm, weights=1, zeros=None,
                reltol=1e-3, feastol=1e-3, abstol=1e-3, pen=None,
                cvg_mono=None, res_sol=None, fac=1, fac_now2=None,
                Xs=None, sYgrid=None, xgrid_qgm=None, Xs_qgm=None,
                sYgrid_qgm=None, info, yorder, ydf, y_knots=None, Ysing, maxit, nXs, nYS,
                lam_vec=None, gam=0, ugrid=None, dedy_min=None, doprimal=False,
                tol_res, bounded=False, Cbound=np.inf, cval=1e-1, algor="ECOS", easy=True, threshold=1e-5, e0mode=False, eta_check=False):

    # Setup
    setup = np.zeros((2, 2))
    setup[0 if algor == "ECOS" else 1][0 if doprimal else 1] = 1

    # Matrix M
    nzeros = len(zeros) if zeros else 0
    if nzeros > 0:
        M = np.zeros((nXs * nYS, nXs * nYS - nzeros))
        M[np.ix_([i for i in range(nXs * nYS) if i not in zeros], range(nXs * nYS - nzeros))] = np.eye(nXs * nYS - nzeros)
    else:
        M = np.eye(nXs * nYS)

    cvg = None
    fac_now = fac
    res0 = 1
    nyg_now = 0
    exit_flag = False

    while cvg is None and tol_res * fac_now < 10 and res0 and not exit_flag:

        print("fac =", fac_now)
        res0 = None

        if nyg != 0:
            pen = True

        if pen and not doprimal:
            print("Switching to primal")
            doprimal = True

        try:
            res0 = gtr_solve(TYX=TYX, tYX=tYX, algor=algor, maxit=maxit, doprimal=doprimal, nXs=nXs, nYS=nYS, Xs=Xs,
                             threshold=threshold, pen=pen, gam=gam, weights=weights, zeros=zeros, lam_vec=lam_vec,
                             sYgrid=sYgrid, cval=cval, reltol=tol_res * fac_now, feastol=feastol, abstol=abstol,
                             bounded=bounded, Cbound=Cbound, beta2=False)
        except:
            pass

        if res0 and (len(res0) <= 1 or np.isnan(res0['llf']) or (not (x.shape[1] == 1 and np.all(x[:, 0] == 1)) and len(np.unique(np.round(res0['eta'], decimals=3))) == 1)):

            print("Trying a different algorithm")
            algor = "SCS" if algor == "ECOS" else "ECOS"
            tol_res = 1e-1 if algor == "SCS" else 1e-4

            try:
                res0 = gtr_solve(TYX=TYX, tYX=tYX, algor=algor, maxit=maxit, doprimal=doprimal, nXs=nXs, nYS=nYS, Xs=Xs,
                                 threshold=threshold, pen=pen, gam=gam, weights=weights, zeros=zeros, lam_vec=lam_vec,
                                 sYgrid=sYgrid, cval=cval, reltol=tol_res * fac_now, feastol=feastol, abstol=abstol,
                                 bounded=bounded, Cbound=Cbound, beta2=False)
            except:
                pass

            if eta_check and not pen and res0 and (np.isnan(res0['llf']) or len(res0) <= 1 or ((not (x.shape[1] == 1 and np.all(x[:, 0] == 1))) and len(np.unique(np.round(res0['eta'], decimals=3))) == 1)):

                exit_flag = True
                print("No algorithm worked")

        if not exit_flag:

            if len(res0) > 1 and eta_check and not exit_flag and gam == 0:

                gtest = grubbs.test(np.log(res0['eta']))
                print("grubbs =", gtest['p.value'])

                if gtest['p.value'] < 0.05:

                    if not pen:

                        doprimal_now = not doprimal
                        res_dual = None

                        try:
                            res_dual = gtr_solve(TYX=TYX, tYX=tYX, algor=algor, maxit=maxit, doprimal=doprimal_now,
                                                 nXs=nXs, nYS=nYS, Xs=Xs, threshold=threshold, pen=pen, gam=gam,
                                                 weights=weights, zeros=zeros, lam_vec=lam_vec, sYgrid=sYgrid,
                                                 cval=cval, reltol=tol_res * fac_now, feastol=feastol, abstol=abstol,
                                                 bounded=bounded, Cbound=Cbound, beta2=False)
                        except:
                            pass

                        if res_dual and len(res_dual) > 1 and not np.isnan(res_dual['llf']):
                            print(abs(res_dual['llf'] - res0['llf']))
                            if abs(res_dual['llf'] - res0['llf']) > 1e-1 or (np.std(res0['eta']) > 1 and abs(res_dual['llf'] - res0['llf']) > 1e-6):
                                print("dgap now =", abs(res_dual['llf'] - res0['llf']))
                                exit_flag = True

                        if len(res_dual) <= 1:
                            exit_flag = True
                            res0 = None

            if not exit_flag:

                if len(res0) > 1 and ((x.shape[1] == 1 and np.all(x[:, 0] == 1)) or len(np.unique(np.round(res0['eta'], decimals=3))) != 1):

                    if cvg_mono is None:

                        cvg = 1
                        nyg_now = nyg
                        print("min dedy now =", np.round(mdedy, decimals=6))

                    if mdedy > np.finfo(float).eps:

                        cvg_mono = 1
                        res_sol = res0
                        fac_now2 = fac_now
                        print("fac.now =", fac_now2)
                        dedy_min = mdedy
                        nyg_now = nyg

            if algor == "ECOS" or nyg == 0:
                fac_now = 11 / tol_res
            else:
                fac_now = 2 * fac * fac_now

    ans = {'res_sol': res_sol, 'res0': res0, 'cvg': cvg, 'nyg_now': nyg_now, 'fac_now2': fac_now2, 'dedy_min': dedy_min, 'algor': algor, 'tol_res': tol_res}
    return ans
def gtr_solve(TYX, tYX, Kscore=0, gam=0, egam=0, lam=0, lam_vec=None, maxit=200, algor="ECOS",
              reltol=1e-04, feastol=1e-04, abstol=1e-04, quiet=False, zeros=None,
              doprimal=False, btarg=0, silent=False, nXs=None, nYS=None, weights=1, cval=0.1,
              pen=None, beta2=False, Xs=None, sYgrid=None, bounded=False, Cbound=1e6, threshold=1e-5):
    M = np.eye(nXs * nYS)
    if zeros is not None:
        M = np.zeros((nXs * nYS, nXs * nYS - len(zeros)))
        M[np.ix_(range(nXs * nYS), np.setdiff1d(range(nXs * nYS), zeros))] = np.eye(nXs * nYS - len(zeros))

    if not doprimal:
        if algor == "SCS" or algor == "ECOS":
            bdim = TYX.shape[1]
            nobs = TYX.shape[0]
            if not quiet:
                print("Problem dimensions are:", nobs, bdim)

            if np.isscalar(Kscore):
                Kscore = np.full((1, 1), Kscore)
            if gam > 0 and np.mean(Kscore) == 0:
                Kscore = np.full((1, 1), gam)
            scorebounds = False
            if np.max(Kscore) > 0:
                scorebounds = True
            Kgauss = np.log(1 / np.sqrt(2 * np.pi))

            e = cp.Variable(nobs)
            h = cp.Variable(nobs)

            def DLLF(e, h):
                llfvec = 0.5 * e ** 2 - cp.log(-h) - 1 + Kgauss
                return cp.sum(llfvec)

            def score(e, h):
                grad = TYX.T @ e + tYX.T @ h
                return grad

            def etafunc(h):
                eta = cp.inv_pos(-h)
                etadev = eta - cp.sum(eta) / nobs
                return eta, etadev

            obj = DLLF(e, h)
            if scorebounds:
                cond1a = score(e, h) >= -Kscore
                cond1b = score(e, h) <= Kscore
                constr = [cond1a, cond1b]
            else:
                cond1 = score(e, h) == 0
                constr = [cond1]

            if bounded:
                cond1 = score(e, h) == 0
                cond2 = cp.norm2(etafunc(h)[1]) <= Cbound
                constr = [cond1, cond2]

            prob = cp.Problem(cp.Minimize(obj), constr)

            if algor == "ECOS":
                arglist = {"max_iters": maxit, "verbose": 1, "reltol": reltol, "feastol": feastol, "abstol": abstol}
                if quiet:
                    arglist["verbose"] = 0
                problem_data = prob.get_problem_data("ECOS")
                if cp.__version__ > "1.1.13":
                    ECOS_dims = cp.reductions.solvers.scs_conif_dgp_extractors.ECOSDimsToSolverDict(
                        problem_data["dims"]).extract()
                    ecos_out1 = cp.reductions.solvers.ecos_conif.ecos_csolve(
                        problem_data["c"], problem_data["G"], problem_data["h"], ECOS_dims, problem_data["A"],
                        problem_data["b"], arglist)
                else:
                    ecos_out1 = cp.reductions.solvers.ecos_conif.ecos_csolve(
                        problem_data["c"], problem_data["G"], problem_data["h"], problem_data["dims"],
                        problem_data["A"], problem_data["b"], arglist)
                result1 = prob.unpack_results("ECOS", ecos_out1)

            if algor == "SCS":
                pd_scs = prob.get_problem_data("SCS")
                arglist2 = {"max_iters": maxit, "eps": reltol}
                if cp.__version__ > "1.1.13":
                    SCS_dims = pd_scs["dims"].extract()
                    scs_out1 = cp.reductions.solvers.scs_conif.scs(
                        pd_scs["A"], pd_scs["b"], pd_scs["c"], SCS_dims, arglist2)
                else:
                    scs_out1 = cp.reductions.solvers.scs_conif.scs(
                        pd_scs["A"], pd_scs["b"], pd_scs["c"], pd_scs["dims"], arglist2)
                result1 = prob.unpack_results("SCS", scs_out1)

            if result1.status in ["solver_error", "unbounded_inaccurate"]:
                ans = {"status": "failed"}
                return ans

            if not quiet:
                print(result1[0].value)

            e = result1[0].value
            h = result1[1].value

            finalscore = score(e, h)
            objval = DLLF(e, h)
            llfvec = np.log(cp.norm(e, 2) * (-1 / h))
            llf = np.sum(llfvec)

            if not scorebounds:
                shadow1 = result1[2].value
                bhat = -shadow1

            if scorebounds:
                shadow1a = result1[2].value
                shadow1b = result1[3].value
                bhat = shadow1a - shadow1b

            ehat = -TYX @ bhat
            etahat = tYX @ bhat

            ans = {"llf": llf, "e": e, "eta": -1 / h, "finalscore": finalscore, "result": result1,
                   "ehat": ehat, "etahat": etahat, "h": h, "objval": objval, "llfvec": llfvec,
                   "bmat": bhat, "this.call": None, "time.CVXR": None}

            return ans

    if doprimal:
        bdim = TYX.shape[1]
        nobs = TYX.shape[0]
        if not quiet:
            print("Problem dimensions are:", nobs, bdim)

        if np.isscalar(Kscore):
            Kscore = np.full((1, 1), Kscore)
        scorebounds = False
        if np.max(Kscore) > 0:
            scorebounds = True

        Kgauss = np.log(1 / np.sqrt(2 * np.pi))

        b = cp.Variable(bdim)

        def LLF(b):
            e = TYX @ b
            dedy = tYX @ b
            llfvec = -0.5 * e ** 2 + cp.log(dedy) + Kgauss
            return cp.sum(llfvec)

        def dedy(b):
            dedy = tYX @ b
            return dedy

        if beta2 or nYS == 2:

            def beta2X(b):
                if isinstance(Xs, list):
                    Xs_now = np.vstack(Xs)
                else:
                    Xs_now = Xs

                Beta = Xs_now @ np.reshape(M @ b, (nXs, nYS))
                return np.min(Beta[:, 1])

        if pen is not None:

            if nYS == 2:
                if nXs > 1:
                    dedy_grid = beta2X
                if nXs == 1:
                    def dedy_grid(b):
                        BetaY = sYgrid @ np.reshape(M @ b, (nYS, nXs))
                        return np.min(BetaY)

            if nYS > 2:

                if nXs > 2:
                    def dedy_grid(b):
                        Xs_now = np.vstack(Xs) if isinstance(Xs, list) else Xs
                        Beta = Xs_now @ np.reshape(M @ b, (nXs, nYS))
                        x = Beta @ sYgrid.T
                        a = -40.
                        return np.min(x)

                if nXs == 2:
                    def dedy_grid(b):
                        BetaY = sYgrid @ np.reshape(M @ b, (nYS, nXs))
                        x1 = BetaY[:, 0] + BetaY[:, 1] * xmin
                        x2 = BetaY[:, 0] + BetaY[:, 1] * xmax
                        return np.min(np.vstack((x1, x2)), axis=0)

                if nXs == 1:
                    def dedy_grid(b):
                        BetaY = sYgrid @ np.reshape(M @ b, (nYS, nXs))
                        return np.min(BetaY)

        def score(e, eta):
            grad = TYX.T @ e + tYX.T @ eta
            return grad

        if Kscore > 0:
            lamx = Kscore * np.asarray(weights)
        elif Kscore == 0 and gam > 0:
            lamspecified = isinstance(lam, np.ndarray) and lam_vec is None
            if not lamspecified:
                lam = np.full((nXs, nYS), lam_vec[6])  # gen
                lam[0, :] = lam_vec[5]  # row1
                lam[:, 0] = lam_vec[3]  # col1
                lam[:, 1] = lam_vec[4]  # col1
                lam[0, 0] = lam_vec[1]  # int
                lam[0, 1] = lam_vec[2]  # int
            if zeros is not None:
                lamx = np.asarray(lam).flatten()[np.setdiff1d(range(nXs * nYS), zeros)] * weights
            else:
                lamx = np.asarray(lam).flatten() * weights
        elif Kscore == 0 and gam == 0:
            lamx = np.zeros(nXs * nYS)

        reg = gam * np.sum(lamx * np.abs(b))
        elastic = egam * cp.norm(b - btarg, 2)
        obj = LLF(b) - reg - elastic

        if pen is None and bounded:
            constraint_condns = cp.norm(b, 2) <= Cbound
            constr = [constraint_condns]
        elif pen is None and beta2:
            constraint_condns = beta2X(b) >= cval
            constr = [constraint_condns]
        elif pen is not None and not bounded and not beta2:
            constraint_condns = dedy_grid(b) >= cval
            constr = [constraint_condns]
        elif pen is not None and bounded and not beta2:
            constraint_condns1 = dedy_grid(b) >= cval
            constraint_condns2 = cp.norm(b, 2) <= Cbound
            constr = [constraint_condns1, constraint_condns2]
        elif pen is not None and not bounded and beta2:
            constraint_condns1 = dedy_grid(b) >= cval
            constraint_condns2 = beta2X(b) >= cval
            constr = [constraint_condns1, constraint_condns2]

        prob = cp.Problem(cp.Maximize(obj), constr)

        if algor == "SCS":
            solver_out1 = prob.solve(solver=algor, verbose=not silent, max_iters=maxit, eps=reltol)
        elif algor == "ECOS":
            solver_out1 = prob.solve(solver=algor, verbose=not silent, MAXIT=maxit, RELTOL=reltol, FEASTOL=feastol,
                                     ABSTOL=abstol)

        if solver_out1.status in ["solver_error", "unbounded", "unbounded_inaccurate", "infeasible"]:
            ans = {"status": "failed"}
            return ans

        bfinal = b.value
        llf = LLF(bfinal)
        e = TYX @ bfinal
        eta = tYX @ bfinal
        finalscore = score(e, eta)
        ehat = e
        etahat = eta
        h = None
        llfvec = np.log(cp.norm(e, 2) * eta)

        ans = {"llf": llf, "e": e, "eta": eta, "finalscore": finalscore, "result": solver_out1,
               "ehat": ehat, "etahat": etahat, "h": h, "llfvec": llfvec, "lamx": lamx, "bmat": bfinal}

        return ans

def beta_check(bmat, Xs, nXs, nYS):
    beta = Xs @ np.reshape(bmat, (nXs, nYS))
    b2min = np.min(beta[:, 1])
    beta2 = beta[:, 1]
    return {"b2min": b2min, "beta2": beta2}
