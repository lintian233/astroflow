# type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import cv2
import multiprocessing
import os
import time

from .dmtime import DmTime
from .io.filterbank import Filterbank
from .dedispered import dedispered_fil_with_dm


class PlotterManager:
    def __init__(self, max_worker=8):
        self.max_worker = max_worker
        self.pool = multiprocessing.Pool(self.max_worker)

    def plot_candidate(self, dmt: DmTime, save_path):
        self.pool.apply_async(plot_candidate, args=(dmt, save_path))

    def plot_spectrogram(self, file_path, candinfo, save_path):
        self.pool.apply_async(plot_spectrogram, args=(file_path, candinfo, save_path))

    def plot_dmtime(self, dmt: DmTime, save_path):
        self.pool.apply_async(plot_dmtime, args=(dmt, save_path))

    def close(self):
        self.pool.close()
        self.pool.join()


def preprocess_img(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img - np.mean(img)) / np.std(img)
    img = cv2.resize(img, (512, 512))

    img = np.clip(img, *np.percentile(img, (0.1, 99.9)))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    img = seaborn.color_palette("mako", as_cmap=True)(img)
    img = img[..., :3]

    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]

    return img


def plot_spectrogram(file_path, candinfo, save_path, dpi=100):
    print(f"Plotting {file_path} with {candinfo}")
    basename = os.path.basename(file_path).split(".")[0]
    fil = Filterbank(file_path)

    dm = candinfo[0]
    toa = candinfo[1]
    freq_start = candinfo[2]
    freq_end = candinfo[3]
    time_size = 0.1
    tstart = toa - time_size
    tend = toa + time_size
    tstart = tstart if tstart > 0 else 0
    tstart = np.round(tstart, 3)
    tend = np.round(tend, 3)
    title = f"{basename}-spectrum-{toa}s-{tstart}s-{tend}s-{dm}pc-cm3"
    spectrum = dedispered_fil_with_dm(fil, tstart, tend, dm, freq_start, freq_end)

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        dpi=dpi,
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
    )
    plt.rcParams["image.origin"] = "lower"
    data = spectrum.data

    vim, vmax = np.percentile(data, [0, 99])
    time_axis = np.linspace(tstart, tend, spectrum.ntimes)
    freq_axis = freq_start + np.arange(spectrum.nchans) * fil.foff
    time_series = data.sum(axis=1)
    axs[0].plot(time_axis, time_series, "k-", linewidth=0.5)
    axs[0].set_ylabel("Integrated Power")
    axs[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axs[0].set_yscale("log")
    axs[0].grid(True, alpha=0.3)
    # add title
    axs[0].set_title(f"{title}")

    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    axs[1].imshow(
        data.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=extent,
        vmin=vim,
        vmax=vmax,
    )
    axs[1].set_ylabel(
        f"Frequency (MHz)\nFCH1={fil.fch1:.3f} MHz, FOFF={fil.foff:.3f} MHz"
    )
    axs[1].set_xlabel(f"Time (s)\nTSAMP={fil.tsamp:.6e}s")

    axs[0].set_xlim(tstart, tend)
    axs[1].set_xlim(tstart, tend)

    plt.subplots_adjust(hspace=0.05, left=0.08, right=0.92)
    plt.savefig(
        f"{save_path}/{title}.png",
        dpi=dpi,
        bbox_inches="tight",
        format="png",
    )
    plt.close()
    print(f"Saved {save_path}/{title}.png")


def plot_dmtime(dmt: DmTime, save_path, dpi=50):
    img = preprocess_img(dmt.data)
    plt.imshow(img)
    plt.title(dmt.__str__())
    # 禁用PNG压缩并保持原始数据精度
    plt.savefig(
        f"{save_path}/{dmt.__str__()}.png",
        dpi=dpi,
        format="png",
        hbox_inches="tight",
        pil_kwargs={"compress_level": 0},
    )
    plt.close()


def plot_candidate(dmt: DmTime, save_path, dpi=150, if_clip=True, if_show=False):
    # start_time = time.time()
    dm_low = dmt.dm_low
    dm_high = dmt.dm_high
    tstart = dmt.tstart
    tend = dmt.tend
    dm_data = dmt.data

    if if_clip:
        dm_data_clip = np.clip(dm_data, *np.percentile(dm_data, (0.02, 99.9)))
    time_axis = np.linspace(tstart, tend, dm_data.shape[1])
    dm_axis = np.linspace(dm_low, dm_high, dm_data.shape[0])

    fig = plt.figure(figsize=(14, 14), dpi=dpi)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[3, 1], height_ratios=[1, 3], hspace=0.05, wspace=0.05
    )

    ax_main = fig.add_subplot(gs[1, 0])
    # 使用imshow并设置正确的坐标范围和方向
    im = ax_main.imshow(
        dm_data_clip,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[time_axis[0], time_axis[-1], dm_axis[0], dm_axis[-1]],
    )
    ax_main.set_xlabel("Time (s)", fontsize=12, labelpad=10)
    ax_main.set_ylabel("DM (pc cm$^{-3}$)", fontsize=12, labelpad=10)

    ax_dm = fig.add_subplot(gs[1, 1], sharey=ax_main)
    dm_sum = np.max(dm_data, axis=1)
    ax_dm.plot(dm_sum, dm_axis, lw=1.5, color="darkblue")
    ax_dm.tick_params(axis="y", labelleft=False)
    ax_dm.grid(alpha=0.3)

    ax_time = fig.add_subplot(gs[0, 0], sharex=ax_main)
    time_sum = np.max(dm_data, axis=0)
    ax_time.plot(time_axis, time_sum, lw=1.5, color="darkred")
    ax_time.tick_params(axis="x", labelbottom=False)
    ax_time.grid(alpha=0.3)

    cax = fig.add_axes([0.25, 0.92, 0.5, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal")

    if if_show:
        plt.show()
    plt.savefig(
        f"{save_path}/{dmt.__str__()}.png",
        dpi=dpi,
        format="png",
        bbox_inches="tight",
    )
    plt.close()
    # print(f"Saved {save_path}/{dmt.__str__()}.png in {time.time() - start_time:.2f}s")
