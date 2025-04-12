from astroflow import dedispered_fil_with_dm, Spectrum, Filterbank
from astroflow.dedispered import dedisperse_spec_with_dm
from astroflow.io.filterbank import Filterbank
from astroflow.io.psrfits import PsrFits

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot dedispersed spectrum")
    parser.add_argument("file_path", type=str, help="Path to the filterbank file")
    parser.add_argument("toa", type=float, help="Start time in seconds")
    parser.add_argument("tband", type=float, default=0.1, help="Time band in seconds")
    parser.add_argument("dm", type=float, help="Dispersion measure in pc cm^-3")
    parser.add_argument("output_path", type=str, help="Path to save the plot")
    parser.add_argument(
        "--freq_start", type=float, default=-1, help="Start frequency in MHz"
    )
    parser.add_argument(
        "--freq_end", type=float, default=-1, help="End frequency in MHz"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to the mask file (optional)",
    )
    return parser.parse_args()


def plot_ded_spectrum(
    file_path, toa, tband, dm, output_path, freq_start=-1, freq_end=-1, mask=None
):
    tstart = toa - tband / 2
    tend = toa + tband / 2
    tstart = max(0, tstart)
    tend = min(tend, 1e10)  # Set a reasonable upper limit for tend
    tstart = round(tstart, 5)
    tend = round(tend, 5)
    os.makedirs(output_path, exist_ok=True)
    basename = os.path.basename(file_path)
    title = f"{basename}-{tstart}s-{tend}s-{dm}pc-cm3"
    origin_data = None

    if file_path.endswith(".fil"):
        origin_data = Filterbank(file_path)
    elif file_path.endswith(".fits"):
        origin_data = PsrFits(file_path)
    else:
        raise ValueError("Unknown file type")

    spectrum = dedisperse_spec_with_dm(
        origin_data, tstart, tend, dm, freq_start, freq_end
    )
    header = origin_data.header()

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        dpi=100,
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
    )
    plt.rcParams["image.origin"] = "lower"

    data = spectrum.data
    if mask is not None:
        print(data.shape)
        data[:, mask[:-1]] = 0

    # vmin, vmax = np.percentile(spectrum.data, [5, 95])
    vmin, vmax = np.percentile(data, [70, 99])
    print(f"vmin: {vmin}, vmax: {vmax}")
    time_axis = np.linspace(tstart, tend, spectrum.ntimes)
    freq_axis = freq_start + np.arange(spectrum.nchans) * header.foff
    time_series = data.sum(axis=1)  # 通道积分
    # 高斯卷积time_series
    # time_series = cv2.GaussianBlur(time_series, (3, 1), 0)
    axs[0].plot(time_axis, time_series, "k-", linewidth=0.5)
    axs[0].set_ylabel("Integrated Power")
    axs[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axs[0].set_yscale("log")
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title(f"{title}")

    # 设置imshow的显示范围和方向
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    axs[1].imshow(
        data.T,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    axs[1].set_ylabel(
        f"Frequency (MHz)\nFCH1={header.fch1:.3f} MHz, FOFF={header.foff:.3f} MHz"
    )
    axs[1].set_xlabel(f"Time (s)\nTSAMP={header.tsamp:.6e}s")

    axs[0].set_xlim(tstart, tend)
    axs[1].set_xlim(tstart, tend)
    plt.subplots_adjust(hspace=0.05, left=0.08, right=0.92)

    source_nane = file_path.split("/")[-1].split(".")[0]
    plt.savefig(
        f"{output_path}/{title}.png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
        format="png",
    )
    print(f"Saved {output_path}/{title}.png")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    print(f"args: {args}")
    mask_file = args.mask
    mask = None
    if mask_file and os.path.exists(mask_file):
        with open(mask_file, "r") as f:
            data = f.read()

        bad_channels = list(map(int, data.split()))

        mask = np.array(bad_channels)

    plot_ded_spectrum(
        args.file_path,
        args.toa,
        args.tband,
        args.dm,
        args.output_path,
        args.freq_start,
        args.freq_end,
        mask=mask,
    )
