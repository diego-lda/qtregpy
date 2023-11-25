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
