from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="qtregpy",
    version="0.0.4",
    author="Sami Stouli, Richard Spady, Xiaoran Liang, Diego Lara",
    author_email="s.stouli@bristol.ac.uk, rspady@jhu.edu, x.liang2@exeter.ac.uk, diegolaradeandres@gmail.com",
    description="A Python package to implement the Quantile Transformation Regression.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diego-lda/qtregpy.git",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
)