import numpy as np
from typing import Dict, Optional, Union

def beta_check(bmat: np.ndarray, Xs: np.ndarray, nXs: int, nYS: int) -> Optional[Dict[str, Union[float, np.ndarray]]]:
    """Performs matrix multiplication and extraction of the minimum value.

    Checks that `bmat` can be reshaped to (nXs, nYS) and that the number of columns
    in `Xs` matches `nXs`. Performs matrix multiplication between `Xs` and reshaped `bmat`,
    then extracts the minimum value from the second column of the resulting matrix.

    Args:
        bmat (np.ndarray): The input matrix to be reshaped and multiplied.
        Xs (np.ndarray): The matrix to be multiplied with `bmat`.
        nXs (int): The number of rows for reshaping `bmat`.
        nYS (int): The number of columns for reshaping `bmat`.

    Returns:
        Optional[Dict[str, Union[float, np.ndarray]]]: A dictionary with the minimum value
        from the second column and the second column of the resulting matrix, or None if
        there was an error.

    """
    try:
        # Check that bmat can be reshaped
        if bmat.size != nXs * nYS:
            print("Error: The total number of elements in bmat doesn't equal nXs * nYS.")
            return None

        # Check that number of columns in Xs equals number of rows in reshaped bmat
        if Xs.shape[1] != nXs:
            print("Error: The number of columns in Xs doesn't match the number of rows in the reshaped bmat.")
            return None

        # Reshape bmat and perform matrix multiplication
        beta = Xs @ np.reshape(bmat, (nXs, nYS))

        # Check that beta has at least two columns
        if beta.shape[1] < 2:
            print("Error: The result of the matrix multiplication doesn't have at least two columns.")
            return None

        # Extract minimum value from the second column of beta and get the second column
        b2min = np.min(beta[:, 1])
        beta2 = beta[:, 1]

        return {"b2min": b2min, "beta2": beta2}

    except Exception as e:
        print("An error occurred: ", e)
        return None