# type: ignore

import astroflow

import numpy as np
import matplotlib.pyplot as plt
import time
import os

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start} seconds")
        return result

    return wrapper

file_path = r"/home/lingh/work/astroflow/tests/qltest.fil"

time_downsample = 1
dm_low = 0
dm_high = 800
freq_start = 1312
freq_end = 1448
dm_step = 1
t_sample = 0.2

timeit_astroflow = timeit(astroflow.dedisper_fil_uint16)


def plot_dedispered_data(dm_data, title, data, tsample, dm_low, dm_high, dpi=100):
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
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


import multiprocessing

def plot_worker(args):
    """多进程绘图的工作函数"""
    dm_matrix, title, tsample, dm_low, dm_high = args
    plot_dedispered_data(dm_matrix, title, None, tsample, dm_low, dm_high)

def draw_dm_times(data, dirname, max_workers=None):
    # 统一小写目录名并创建目录
    dirname = dirname.lower()
    os.makedirs(dirname, exist_ok=True)
    
    # 准备多进程参数
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
    
    # 自动设置最大进程数（不超过CPU核心数）
    if max_workers is None:
        max_workers = min(len(plot_args), os.cpu_count() - 64 or 4)
    
    # 使用进程池并行执行
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

if __name__ == "__main__":
    dedispered_dir("/data/QL/rigol/7310_FRB20201124A_25-01-22_14-43-33")
