import pytest
import numpy as np
import math

import qtregpy as qtr

def test_calc_loglike():
    # define your matrices
    b = np.array([1, 2, 3])
    TYX = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tYX = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # calculate the expected result manually
    Kgauss = math.log(1 / math.sqrt(2 * math.pi))
    e = np.matmul(TYX, b)
    dedy = np.matmul(tYX, b)
    llfvec = -.5 * e ** 2 + np.log(dedy) + Kgauss
    expected_result = np.sum(llfvec)

    # compare the expected result to the result from your function
    assert np.isclose(qtr.calc_loglike(b, TYX, tYX), expected_result)