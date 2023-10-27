import pytest
import numpy as np
import math

from qtregpy import compute

def test_print_name(capfd):
    # Exercise
    compute.print_name("John")  # Call the function

    # Verify
    out, err = capfd.readouterr()  # Capture the print output
    assert out == "Hello, John!\n"  # Verify the output

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
    assert np.isclose(compute.calc_loglike(b, TYX, tYX), expected_result)