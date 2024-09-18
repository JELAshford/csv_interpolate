from scipy.ndimage import convolve
from numba import njit
import numpy as np
import timeit


@njit
def interpolate_at_nans(matrix: np.ndarray):
    """Iterate over all 'nan' positions in the input matrix and replace those
    values with the average of all non-'nan' values in the Von Neumann neighbourhood
    around it.

    Args:
        matrix (np.ndarray): Input 2D matrix

    Returns:
        out_matrix (np.ndarray): Output 2D matrix with interpolated values
    """
    out_matrix = matrix.copy()
    height, width = matrix.shape
    # Iterate over all 'nan' locations
    nan_locations = np.argwhere(np.isnan(matrix))
    for y, x in nan_locations:
        # Count number of and sum all valid neighbour values
        neighbour_sum, count = 0, 0
        for y_off, x_off in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            y_prime = y + y_off
            x_prime = x + x_off
            if (0 <= y_prime < height) and (0 <= x_prime < width):
                neighbour_sum += matrix[y_prime, x_prime]
                count += 1
        # Store average in the output if enough values available
        if count > 0:
            out_matrix[y, x] = neighbour_sum / count
    return out_matrix


def interpolate_convolve(matrix):
    """Convolve over the input array to count the number and sum of neighbours in
    given neighbourhood, replacing all 'nan's with the neighbour average.

    Args:
        matrix (np.ndarray): Input 2D matrix

    Returns:
        matrix (np.ndarray): Output 2D matrix with interpolated values
    """
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
    IN_MATRIX = np.loadtxt("rsc/input_test_data.csv", delimiter=",")

    # Run functions once for warmup/JIT compile
    warmup_matrix = np.tile(IN_MATRIX, (5, 5))
    _ = interpolate_at_nans(warmup_matrix)
    _ = interpolate_convolve(warmup_matrix)

    # Run `timeit` testing on larger matrices
    run_params = {"globals": locals(), "number": 1000}
    test_matrix = np.tile(IN_MATRIX, (50, 50))
    iterated_time = timeit.timeit("interpolate_at_nans(test_matrix)", **run_params)
    convolve_time = timeit.timeit("interpolate_convolve(test_matrix)", **run_params)
    print(f"Numba: {iterated_time:.2f}s")
    print(f"Convolve: {convolve_time:.2f}s")
    print(f"Numba speedup: {convolve_time/iterated_time:.2f}x")
