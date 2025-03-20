# type: ignore

import astroflow

import numpy as np
import matplotlib.pyplot as plt
import time
import os

import cv2

from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
from astroflow.utils import draw_dm_times

time_downsample = 1
dm_low = 0
dm_high = 800
freq_start = 1050
freq_end = 1448
dm_step = 0.5
t_sample = 0.5


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start} seconds")
        return result

    return wrapper


timeit_astroflow = timeit(astroflow.dedispered_fil)


def dedispered_dir(dir):
    all_files = os.listdir(dir)
    print(f"Processing directory {dir}")
    base_dir = os.path.basename(dir)

    for file in all_files:
        if not file.endswith(".fil"):
            continue

        file_stem = os.path.splitext(file)[0]
        file_dir = os.path.join(base_dir, file_stem).lower()
        print(f"checking {file_dir}")
        if os.path.exists(file_dir):
            print(f"跳过已处理文件: {file} (输出目录 {file_dir} 已存在)")
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
        )
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
    )
    # data.__class__ = astroflow.DedispersedData
    print(data.__class__)
    print("save:", "ql/" + file.split(".")[0].split("/")[-1])
    draw_dm_times(data, "ql/" + file.split(".")[0].split("/")[-1])


if __name__ == "__main__":
    dedispred_single("/home/lingh/work/astroflow/tests/FRB20241124A.fil")
