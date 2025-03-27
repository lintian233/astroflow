from astroflow import dedispered_fil_with_dm, Spectrum, Filterbank

import numpy as np
import matplotlib.pyplot as plt
import os

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot dedispersed spectrum")
    parser.add_argument("file_path", type=str, help="Path to the filterbank file")
    parser.add_argument("tstart", type=float, help="Start time in seconds")
    parser.add_argument("tend", type=float, help="End time in seconds")
    parser.add_argument("dm", type=float, help="Dispersion measure in pc cm^-3")
    parser.add_argument("output_path", type=str, help="Path to save the plot")
    parser.add_argument(
        "--freq_start", type=float, default=-1, help="Start frequency in MHz"
    )
    parser.add_argument(
        "--freq_end", type=float, default=-1, help="End frequency in MHz"
    )
    return parser.parse_args()


def plot_ded_spectrum(
    file_path, tstart, tend, dm, output_path, freq_start=-1, freq_end=-1
):
    basename = os.path.basename(file_path)
    title = f"{basename}-{tstart}s-{tend}s-{dm}pc-cm3"
    fil = Filterbank(file_path)

    spectrum = dedispered_fil_with_dm(fil, tstart, tend, dm, freq_start, freq_end)

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        dpi=100,
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
    )
    plt.rcParams["image.origin"] = "lower"

    vmin, vmax = np.percentile(spectrum.data, [0, 99])
    print(f"vmin: {vmin}, vmax: {vmax}")
    data = spectrum.data
    time_axis = np.linspace(tstart, tend, spectrum.ntimes)
    freq_axis = freq_start + np.arange(spectrum.nchans) * fil.foff
    time_series = data.sum(axis=1)  # 通道积分
    axs[0].plot(time_axis, time_series, "k-", linewidth=0.5)
    axs[0].set_ylabel("Integrated Power")
    axs[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axs[0].set_yscale("log")
    axs[0].grid(True, alpha=0.3)

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
        f"Frequency (MHz)\nFCH1={fil.fch1:.3f} MHz, FOFF={fil.foff:.3f} MHz"
    )
    axs[1].set_xlabel(f"Time (s)\nTSAMP={fil.tsamp:.6e}s")

    axs[0].set_xlim(tstart, tend)
    axs[1].set_xlim(tstart, tend)
    plt.subplots_adjust(hspace=0.05, left=0.08, right=0.92)
    plt.savefig(
        f"{output_path}/{title}.png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
        format="png",
        pil_kwargs={"compress_level": 0},
    )
    print(f"Saved {output_path}/{title}.png")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    plot_ded_spectrum(
        args.file_path,
        args.tstart,
        args.tend,
        args.dm,
        args.output_path,
        args.freq_start,
        args.freq_end,
    )
