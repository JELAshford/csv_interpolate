from .interp_methods import interpolate_at_nans, interpolate_convolve
import numpy as np
import click


INTERP_METHODS = {
    "convolve": interpolate_convolve,
    "iterated": interpolate_at_nans,
}
METHODS = f"`{'`, `'.join(list(INTERP_METHODS.keys()))}`"


@click.command()
@click.option("--csv_path", help="Path to file you want to interpolate")
@click.option(
    "--method", default="convolve", help=f"Algorithm to use, one of: {METHODS}"
)
@click.option(
    "--save_path", default="output.csv", help="Path to save interpolated matrix to"
)
def interpolate(csv_path: str, method: str = "iterated", save_path: str = "output.csv"):
    """Interpolate a 2D matrix, provided as a '.csv' file, so all `nan` values are replaced
    with the average of their neighbours."""
    # Load matrix
    in_matrix = np.loadtxt(csv_path, delimiter=",")
    # Interpolate with chosen method
    out_matrix = INTERP_METHODS[method](in_matrix)
    # Save to file
    np.savetxt(save_path, out_matrix, delimiter=",", fmt="%.6f")


if __name__ == "__main__":
    interpolate()
