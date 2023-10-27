import numpy as np
import pytest
from qtregpy import gtreg_utils

def test_beta_check():
    """Tests the beta_check function in utils.py.

    This test covers the following cases:
    1. Successful operation with valid input.
    2. Failure due to incompatible shapes between Xs and bmat.
    3. Failure due to insufficient number of columns in the result.

    """
    # Case 1: Successful operation with valid input
    Xs = np.array([[1, 2, 3], [4, 5, 6]])
    bmat = np.array([1, 2, 3, 4, 5, 6])
    result = gtreg_utils.beta_check(bmat, Xs, 3, 2)
    assert result is not None
    assert result["b2min"] == 28.0
    assert np.array_equal(result["beta2"], np.array([28, 64]))

    # Case 2: Failure due to incompatible shapes between Xs and bmat
    Xs = np.array([[1, 2, 3], [4, 5, 6]])
    bmat = np.array([1, 2, 3, 4])
    result = gtreg_utils.beta_check(bmat, Xs, 2, 2)
    assert result is None

    # Case 3: Failure due to insufficient number of columns in the result
    Xs = np.array([[1, 2], [3, 4]])
    bmat = np.array([1, 2])
    result = gtreg_utils.beta_check(bmat, Xs, 2, 1)
    assert result is None
    