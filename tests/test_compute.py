import pytest
from gtregpy import compute

def test_print_name(capfd):
    # Exercise
    compute.print_name("John")  # Call the function

    # Verify
    out, err = capfd.readouterr()  # Capture the print output
    assert out == "Hello, John!\n"  # Verify the output