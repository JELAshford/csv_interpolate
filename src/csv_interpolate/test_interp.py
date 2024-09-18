from .interp_methods import interpolate_at_nans, interpolate_convolve
import numpy as np


IN_MATRIX = np.loadtxt("rsc/input_test_data.csv", delimiter=",")
OUT_MATRIX = np.loadtxt("rsc/interpolated_test_data.csv", delimiter=",")


def test_iterated_interp_correct():
    out_matrix = interpolate_at_nans(IN_MATRIX)
    assert np.isclose(out_matrix, OUT_MATRIX).all()


def test_conv_interp_correct():
    out_matrix = interpolate_convolve(IN_MATRIX)
    assert np.isclose(out_matrix, OUT_MATRIX).all()
