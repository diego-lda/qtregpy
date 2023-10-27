from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="qtregpy",
    version="0.1.0",
    author="Sami Stouli, Richard Spady, Xiaoran Liang, Diego Lara",
    author_email="s.stouli@bristol.ac.uk, rspady@jhu.edu, x.liang2@exeter.ac.uk, diegolaradeandres@gmail.com",
    description="A Python package to implement the Quantile Transformation Regression.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diego-lda/qtregpy",
    download_url="https://github.com/diego-lda/qtregpy/archive/refs/tags/0.0.4.tar.gz",
    keywords=['DISTRIBUTION', 'DENSITY', 'ECONOMETRICS', 'ESTIMATION'],
    packages=find_packages(),
    requires=[
        'cvxpy',
        'ecos',
        'iniconfig',
        'numpy',
        'osqp',
        'packaging',
        'pluggy',
        'pytest',
        'qdldl',
        'scipy',
        'scs'
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
)