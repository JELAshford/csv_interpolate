# CSV Interpolation Tool

Python command line interface (CLI) tool to replace 'nan's in '.csv' files with the average of their neighbours.

## Installation

This package can be installed into your local `Python` environment using `pip` or into an isolated virtual environment with `pipx`. During development I used the [`uv` Python project manager](https://github.com/astral-sh/uv), which allows fast environment creation/management. With `uv` installed you can get the CLI in the same state as I had during development by runnning these commands in the project root directory:

```bash
uv sync
source .venv/bin/activate
```

and all tests can be run with:

```bash
uv pip install pytest
pytest
```

## Usage

Installation of this packages exposes a CLI that can be invoked with the command `interpolate`.
Running `interpolate --help` produces:

```bash
 interpolate --help
Usage: interpolate [OPTIONS]

  Interpolate a 2D matrix, provided as a '.csv' file, so all `nan` values are
  replaced with the average of their neighbours.

Options:
  --csv_path TEXT   Path to file you want to interpolate
  --method TEXT     Algorithm to use, one of: `convolve`, `iterated`
  --save_path TEXT  Path to save interpolated matrix to
  --help            Show this message and exit.
```

## Method Notes

The simplest and most flexible approach to solving this problem is through convolution, using the `scipy` convolution method on a `numpy` array containing the matrix data. This method is the CLI default and the most generalisable, as it is easy to update the code with different neighbourhood kernels or add features such as 'wrapping' around the edges of the matrix.

I have also written an alternative method that individually finds and replaces the 'nan' values in the matrix, which I have accelerated with JIT compilation throught the `numba` library. This method is ~5-10x faster than the full convolutional method, likely as the number of 'nan' values is low in the examples resulting in less work to do compared to the full array convolution. Additionally, this method is less flexible and requires the installation of more packages.

To prioritise generalisability the convolution method is the default solution. However, it would be interesting to discuss usual use-cases with other members of the team and benchmark the method on arrays of different sizes and sparsity of 'nan's to decide whether a change of method would be more effective. Additionally, if the performance of the algorithm depended heavily of the sparsity of the input matrix, this could first be calculated then the most appropriate algorithm applied.

The benchmark timings for each method can be seen by running:

```bash
uv run src/csv_interpolate/interp_methods.py
```
