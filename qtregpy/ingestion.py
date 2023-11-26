import csv
import numpy as np

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

def data_norm(x: np.ndarray, mean: float = 0, sd: float = 1, log: bool = False) -> np.ndarray:
    """
    Data Normalization

    This function normalizes the data. It applies a transformation so that the data
    has a specified mean and standard deviation.

    Args:
        x (np.ndarray): The data to be normalized.
        mean (float, optional): The value to center on. Defaults to 0.
        sd (float, optional): The standard deviation to target. Defaults to 1.
        log (bool, optional): Whether to apply a logarithmic transformation to the data. Defaults to False.

    Returns:
        np.ndarray: A normalized version of the data.

    Examples:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> normalized_data = data_norm(data, mean=0, sd=1)
        >>> print(normalized_data)
        >>> log_normalized_data = data_norm(data, mean=0, sd=1, log=True)
        >>> print(log_normalized_data)
    """
    if log:
        x = np.log(x)

    ans = 0.3989422804014327 * np.exp(-0.5 * ((x - mean) / sd) ** 2) / sd
    return ans