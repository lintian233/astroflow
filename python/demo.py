# type: ignore

import astroflow

import numpy as np
import matplotlib.pyplot as plt
import time
import os


time_downsample = 1
dm_low = 0
dm_high = 2000
freq_start = 1131
freq_end = 1464
dm_step = 1
t_sample = 1

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start} seconds")
        return result

    return wrapper


timeit_astroflow = timeit(astroflow.dedisper_fil_uint8)


def plot_dedispered_data(dm_data, title, data, tsample, dm_low, dm_high, dpi=100):
    time_axis = np.linspace(0, tsample * dm_data.shape[1], dm_data.shape[1])
    dm_axis = np.linspace(dm_low, dm_high, dm_data.shape[0])

    plt.figure(figsize=(12, 8), dpi=dpi)
    X, Y = np.meshgrid(time_axis, dm_axis)
    im = plt.pcolormesh(X, Y, dm_data, shading="auto", cmap="viridis")
    #将dm_data中的0值替换为nan
    dm_data = np.where(dm_data == 0, np.mean(dm_data), dm_data)
    print(np.percentile(dm_data,99), np.max(dm_data))
    plt.xlabel("Time (tsample)", fontsize=12, labelpad=10)
    plt.ylabel("DM (pc cm$^{-3}$)", fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, pad=15)
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Signal Strength (arb. units)", rotation=270, labelpad=15)
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


import multiprocessing

def plot_worker(args):
    dm_matrix, title, tsample, dm_low, dm_high = args
    plot_dedispered_data(dm_matrix, title, None, tsample, dm_low, dm_high)

def draw_dm_times(data, dirname, max_workers=None):
    dirname = dirname.lower()
    os.makedirs(dirname, exist_ok=True)
    
    filename = data.filname
    basename = os.path.splitext(os.path.basename(filename))[0]
    plot_args = [
        (
            dm.reshape(data.shape[0], data.shape[1]),
            f"{dirname}/{basename}-DM-{data.dm_low}-{data.dm_high}-{data.dm_step}-Trial-{idx+1}",
            data.tsample,
            data.dm_low,
            data.dm_high
        )
        for idx, dm in enumerate(data.dm_times)
    ]
    
    if max_workers is None:
        max_workers = min(len(plot_args), os.cpu_count() - 64 or 4)
    
    with multiprocessing.Pool(processes=max_workers) as pool:
        pool.map(plot_worker, plot_args)

def dedispered_dir(dir):
    all_files = os.listdir(dir)
    for file in all_files:
        if not file.endswith(".fil"):
            continue
        file_path = os.path.join(dir, file)
        print(f"Processing {file_path}")
        data = timeit_astroflow(
            file_path,
            dm_low,
            dm_high,
            freq_start,
            freq_end,
            dm_step,
            time_downsample,
            t_sample,
            njobs=120,
        )
        file_dir = "ql/" + file.split(".")[0]
        draw_dm_times(data, file_dir)


def dedispred_single(file):
    data = timeit_astroflow(
        file,
        dm_low,
        dm_high,
        freq_start,
        freq_end,
        dm_step,
        time_downsample,
        t_sample,
        njobs=120,
    )
    #data.__class__ = astroflow.DedispersedData
    print(data.__class__)
    data0 = data.dm_times[0]
    data0 = data0.reshape(data.shape[0], data.shape[1])
    data1 = data.dm_times[1]
    data1 = data1.reshape(data.shape[0], data.shape[1])
    file = os.path.basename(file)
    draw_dm_times(data, "ql/" + file.split(".")[0])

if __name__ == "__main__":
   dedispred_single("/home/lingh/work/astroflow/tests/FRB20171116.fil")
