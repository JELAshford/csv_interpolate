from scipy.ndimage import convolve
from numba import njit
import numpy as np
import timeit


@njit
def interpolate_at_nans(matrix):
    out_matrix = matrix.copy()
    height, width = matrix.shape
    nan_locations = np.argwhere(np.isnan(matrix))
    for y, x in nan_locations:
        neighbour_sum, count = 0, 0
        for y_off, x_off in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            y_prime = y + y_off
            x_prime = x + x_off
            if (0 <= y_prime < height) and (0 <= x_prime < width):
                neighbour_sum += matrix[y_prime, x_prime]
                count += 1
        out_matrix[y, x] = neighbour_sum / count
    return out_matrix


def interpolate_convolve(matrix):
    neighbourhood = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    nan_mask = np.isnan(matrix)
    # Count neighbour sum and number of active neighbours
    sum_neighbours = convolve(np.nan_to_num(matrix), neighbourhood, mode="constant")
    num_neighbours = convolve((~nan_mask).astype(int), neighbourhood, mode="constant")
    # Calculate averages
    neighbour_averages = sum_neighbours / num_neighbours
    matrix[nan_mask] = neighbour_averages[nan_mask]
    return matrix


if __name__ == "__main__":
    # Run basic benchmarking
    DATA_DIR = "../interpolation/example_data/"
    IN_MATRIX = np.loadtxt(f"{DATA_DIR}input_test_data.csv", delimiter=",")
    SAMPLE_OUT_MATRIX = np.loadtxt(
        f"{DATA_DIR}/interpolated_test_data.csv", delimiter=","
    )

    # Run functions once for warmup/JIT compile
    warmup_matrix = np.tile(IN_MATRIX, (5, 5))
    _ = interpolate_at_nans(warmup_matrix)
    _ = interpolate_convolve(warmup_matrix)

    # Run `timeit` testing on larger matrices
    run_params = {"globals": locals(), "number": 1000}
    test_matrix = np.tile(IN_MATRIX, (20, 20))
    print(timeit.timeit("interpolate_at_nans(test_matrix)", **run_params))
    print(timeit.timeit("interpolate_convolve(test_matrix)", **run_params))
