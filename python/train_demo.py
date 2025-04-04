import astropy.io.fits as fits
import numpy as np
import time


def timeit(func):
    """
    Decorator to time a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


@timeit
def read_fits_file(file_path):
    """
    Read a FITS file and return the data and header.

    Parameters:
    file_path (str): Path to the FITS file.

    Returns:
    tuple: Data and header from the FITS file.
    """
    with fits.open(file_path) as hdul:
        data = data = hdul[1].data["DATA"][:, :, 0, :, 0]
        header = hdul[1].header
    return data, header


def plot_data(data):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(30, 8), dpi=100)
    vmin, vmax = np.percentile(data, (0.1, 95))
    plt.imshow(
        data.T, cmap="viridis", origin="lower", aspect="auto", vmin=vmin, vmax=vmax
    )
    plt.tight_layout()
    plt.savefig("test.png", dpi=100, bbox_inches="tight")


def main():

    # Example FITS file path
    start_time = time.time()
    fits_file_path = r"/data/QL/lingh/FAST_FRB_DATA/FRB20201124_0001.fits"
    tdata = 2.700315393
    # Read the FITS file
    data, header = read_fits_file(fits_file_path)
    print(f"NPOL: {header['NPOL']}")
    print(f"NSBLK: {header['NSBLK']}")
    print(f"TBIN: {header['TBIN']}")
    chan_bw = header["CHAN_BW"]  # [MHz] channel bandwidth
    nchan = header["NCHAN"]  # Nr of channels
    nbits = header["NBITS"]  # Nr of bits per sample
    nsblk = header["NSBLK"]  # [samples]
    tbin = header["TBIN"]  # [s] Time per bin or sample

    tsamp = header["TBIN"]
    tidx = int(tdata / tsamp)
    first_idx = tidx // header["NSBLK"]
    # +0.5s -0.5s
    t_left = tdata - 0.5
    t_right = tdata + 0.5
    left_idx = int(t_left / tsamp)
    right_idx = int(t_right / tsamp)
    end_time = time.time()
    data = data.reshape(-1, 4096)
    data = data[left_idx:right_idx, :]

    print(
        f"Time taken to read and process the FITS file: {end_time - start_time:.4f} seconds"
    )
    print(data.shape)
    print(f"tidx: {tidx}, first_idx: {first_idx}")
    plot_data(data)


if __name__ == "__main__":
    main()
