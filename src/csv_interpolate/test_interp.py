import numpy as np


def interp_iterated(matrix):
    out_matrix = matrix.copy()
    height, width = matrix.shape
    nan_locations = np.argwhere(np.isnan(matrix))
    for y, x in nan_locations:
        neighbour_coords = [
            (y, x - 1),
            (y, x + 1),
            (y - 1, x),
            (y + 1, x),
        ]
        neighbour_vals = []
        for y_prime, x_prime in neighbour_coords:
            if (0 <= y_prime < height) and (0 <= x_prime < width):
                neighbour_vals.append(matrix[y_prime, x_prime])
        out_matrix[y, x] = np.mean(neighbour_vals)
    return out_matrix


def interp_fullshift(matrix):
    nan_mask = np.isnan(matrix)
    num_matrix = np.nan_to_num(matrix.copy())
    sum_matrix = np.zeros_like(matrix)
    count_matrix = np.zeros_like(matrix)

    # Below
    sum_matrix[:-1, :] += num_matrix[1:, :]
    count_matrix[:-1, :] += ~nan_mask[1:, :]
    # Above
    sum_matrix[1:, :] += num_matrix[:-1, :]
    count_matrix[1:, :] += ~nan_mask[:-1, :]
    # Right
    sum_matrix[:, :-1] += num_matrix[:, 1:]
    count_matrix[:, :-1] += ~nan_mask[:, 1:]
    # Left
    sum_matrix[:, 1:] += num_matrix[:, :-1]
    count_matrix[:, 1:] += ~nan_mask[:, :-1]

    count_matrix[count_matrix == 0] = 1

    out_matrix = matrix.copy()
    out_matrix[nan_mask] = (sum_matrix / count_matrix)[nan_mask]

    return out_matrix


def test_iterated_interp_correct():
    out_matrix = interp_iterated(IN_MATRIX)
    assert np.isclose(out_matrix, SAMPLE_OUT_MATRIX).all()


def test_shifting_interp_correct():
    out_matrix = interp_fullshift(IN_MATRIX)
    assert np.isclose(out_matrix, SAMPLE_OUT_MATRIX).all()


DATA_DIR = "../interpolation/example_data/"

# Load in matrix
IN_MATRIX = np.loadtxt(f"{DATA_DIR}input_test_data.csv", delimiter=",")
SAMPLE_OUT_MATRIX = np.loadtxt(f"{DATA_DIR}/interpolated_test_data.csv", delimiter=",")

# # Save matrix
# np.savetxt("out_matrix.txt", out_matrix, delimiter=",", fmt="%.6f")
