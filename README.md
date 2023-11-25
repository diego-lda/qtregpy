# qtregpy

This package provides the software tools to implement Quantile Transformation Regression introduced in
[Spady and Stouli (2020)](https://arxiv.org/pdf/2011.06416.pdf).

With the tools in this package it is possible to obtain inference and estimation results for conditional distribution,
quantile and density functions implied by flexible Gaussian representations.

For further details, please refer to the original text.

## Installation

You can install gtregpy with pip directly from PyPI.

`pip install qtregpy`

## Usage

Here's a simple example of how to use gtregpy:

```python
import qtregpy as qtr

mel_data_path = 'filepath/melbeourne.csv'

x, y = qtr.load_mel_data(mel_data_path)
mel_answer = qtr.compute_basic(x, y)
```

## Documentation

You can find more detailed documentation for each function in docstrings.

## Testing

To run the tests, use the following command:

```python
pytest tests/
```

## Contributing

We welcome contributions! Please see the [Contribution Guide](CONTRIBUTING.md) file for details on how to contribute.

## License

This package is licensed under the MIT license. See the LICENSE file for details.
