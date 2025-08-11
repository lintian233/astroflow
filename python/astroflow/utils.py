import time
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start} seconds")
        return result

    return wrapper


class SingleDmConfig:
    def __init__(self, dm, freq_start, freq_end, t_sample):
        self.dm = dm
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.t_sample = t_sample

class Config:
    def __init__(
        self,
        dm_low,
        dm_high,
        freq_start,
        freq_end,
        dm_step,
        time_downsample,
        t_sample,
        confidence=0.5,
    ):
        self.dm_low = dm_low
        self.dm_high = dm_high
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.dm_step = dm_step
        self.time_downsample = time_downsample
        self.t_sample = t_sample
        self.confidence = confidence

    def __str__(self):
        info = f"{self.dm_low}_{self.dm_high}_{self.freq_start}MHz_{self.freq_end}MHz_{self.dm_step}_{self.t_sample}s"
        return info

    def __repr__(self):
        return self.__str__()


def plot_dedispered_data(dm_data, title, tsample, dm_low, dm_high, dpi=100):
    time_axis = np.linspace(0, tsample * dm_data.shape[1], dm_data.shape[1])
    dm_axis = np.linspace(dm_low, dm_high, dm_data.shape[0])
    plt.figure(figsize=(12, 8), dpi=dpi)
    X, Y = np.meshgrid(time_axis, dm_axis)
    im = plt.pcolormesh(X, Y, dm_data, shading="auto", cmap="viridis")
    plt.xlabel("Time (tsample)", fontsize=12, labelpad=10)
    plt.ylabel("DM (pc cm$^{-3}$)", fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, pad=15)
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Signal Strength (arb. units)", rotation=270, labelpad=15)
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=dpi)
    plt.close()


def plot_worker(args):
    dm_matrix, title, tsample, dm_low, dm_high = args
    plot_dedispered_data(dm_matrix, title, tsample, dm_low, dm_high)


def draw_dm_times(data, dirname, max_workers=None):
    dirname = dirname.lower()
    os.makedirs(dirname, exist_ok=True)

    filename = data.filname
    basename = os.path.splitext(os.path.basename(filename))[0]
    info = f"{dirname}/{basename}-DM-{data.dm_low}-{data.dm_high}-{data.dm_step}-tsample-{data.tsample}"
    plot_args = [
        (
            dm.reshape(data.shape[0], data.shape[1]),
            info + str(idx),
            data.tsample,
            data.dm_low,
            data.dm_high,
        )
        for idx, dm in enumerate(data.dm_times)
    ]

    if max_workers is None:
        max_workers = min(len(plot_args), os.cpu_count() or 4)

    with multiprocessing.Pool(processes=max_workers) as pool:
        pool.map(plot_worker, plot_args)
