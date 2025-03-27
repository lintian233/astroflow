# type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import cv2
import multiprocessing

from .dmtime import DmTime


class PlotterManager:
    def __init__(self, max_worker=32):
        self.max_worker = max_worker
        self.pool = multiprocessing.Pool(self.max_worker)

    def plot_candidate(self, dmt: DmTime, save_path):
        self.pool.apply_async(plot_candidate, args=(dmt, save_path))

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


def plot_dmtime(dmt: DmTime, save_path, dpi=50):
    img = preprocess_img(dmt.data)
    plt.imshow(img)
    plt.title(dmt.__str__())
    plt.savefig(f"{save_path}/{dmt.__str__()}.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_candidate(dmt: DmTime, save_path, dpi=150, if_clip=True, if_show=False):
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
    X, Y = np.meshgrid(time_axis, dm_axis)
    im = ax_main.pcolormesh(X, Y, dm_data_clip, shading="auto", cmap="viridis")
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

    plt.tight_layout()
    if if_show:
        plt.show()
    plt.savefig(f"{save_path}/{dmt.__str__()}.png", dpi=dpi, bbox_inches="tight")
    plt.close()
