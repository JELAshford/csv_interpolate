from .interp_methods import interpolate_at_nans, interpolate_convolve
import numpy as np


DATA_DIR = "../interpolation/example_data/"
IN_MATRIX = np.loadtxt(f"{DATA_DIR}input_test_data.csv", delimiter=",")
SAMPLE_OUT_MATRIX = np.loadtxt(f"{DATA_DIR}/interpolated_test_data.csv", delimiter=",")


# Acceptance tests: do each of these methods complete the sample data?
def test_iterated_interp_correct():
    out_matrix = interpolate_at_nans(IN_MATRIX)
    assert np.isclose(out_matrix, SAMPLE_OUT_MATRIX).all()


def test_conv_interp_correct():
    out_matrix = interpolate_convolve(IN_MATRIX)
    assert np.isclose(out_matrix, SAMPLE_OUT_MATRIX).all()


# Unit testing:
