import torch
from centernet_utils import get_res
from centernet_model import centernet
import astroflow
import time
import os
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

import seaborn

from astroflow import dedispered_fil, Config
from astroflow.utils import timeit
import astroflow.utils as utils


# time_downsample = 1
# dm_low = 0
# dm_high = 800
# freq_start = 1130
# freq_end = 1350
# dm_step = 0.5
# t_sample = 0.5


def preprocess_img(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img - np.mean(img)) / np.std(img)
    img = cv2.resize(img, (512, 512))

    img = np.clip(img, *np.percentile(img, (0.1, 99.9)))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # img = plt.get_cmap("mako")(img)
    img = seaborn.color_palette("mako", as_cmap=True)(img)
    img = img[..., :3]

    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]

    return img  # data.__class__ = astroflow.DedispersedData


def dedispred_single_file(
    file,
    config: Config,
):
    timeit_astroflow = timeit(astroflow.dedispered_fil)
    data = timeit_astroflow(
        file,
        config.dm_low,
        config.dm_high,
        config.freq_start,
        config.freq_end,
        config.dm_step,
        config.time_downsample,
        config.t_sample,
    )
    dmt_data = [dmt.reshape(data.shape[0], data.shape[1]) for dmt in data.dm_times]
    dmt_data = np.array(dmt_data)
    return dmt_data


def plot_img(img, filepath, config: Config, title=None, dpi=50):
    plt.imshow(img)
    plt.title(config.__str__())
    plt.savefig(f"{filepath}/T_{title}_{config.__str__()}.png", dpi=dpi)


def plot_worker(args):
    img = preprocess_img(args[0])
    plot_img(img, args[1], args[2], args[3])


def plot_all_img(dmt_data, dir_path, config: Config):
    plot_args = [
        (
            dmt,
            f"{dir_path}",
            config,
            idx,
        )
        for idx, dmt in enumerate(dmt_data)
    ]

    max_workers = min(len(plot_args), os.cpu_count() or 4)

    with multiprocessing.Pool(processes=max_workers) as pool:
        pool.map(plot_worker, plot_args)


def plot_candidate(
    dm_data, filepath, config: Config, title=None, dpi=150, if_clip=True
):
    dm_low = config.dm_low
    dm_high = config.dm_high
    tsample = config.t_sample
    if if_clip:
        dm_data = np.clip(dm_data, *np.percentile(dm_data, (0.02, 99.9)))
    time_axis = np.linspace(0, tsample * dm_data.shape[1], dm_data.shape[1])
    dm_axis = np.linspace(dm_low, dm_high, dm_data.shape[0])
    plt.figure(figsize=(12, 8), dpi=dpi)
    X, Y = np.meshgrid(time_axis, dm_axis)
    im = plt.pcolormesh(X, Y, dm_data, shading="auto", cmap="viridis")
    plt.xlabel("Time (tsample)", fontsize=12, labelpad=10)
    plt.ylabel("DM (pc cm$^{-3}$)", fontsize=12, labelpad=10)
    plt.title(f"T_{title}_{config.__str__()}", fontsize=14, pad=15)
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Signal Strength (arb. units)", rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig(f"{filepath}/T_{title}_{config.__str__()}.png", dpi=dpi)
    plt.close()


def detect_frb(file, config: Config, output_dir, model, device):
    detect_dir = os.path.join(output_dir, "detect").lower()
    file_basename = file.split("/")[-1].split(".")[0].lower()
    save_path = os.path.join(output_dir, file_basename).lower()

    os.makedirs(detect_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    dmt_data = dedispred_single_file(file, config)
    plot_all_img(dmt_data, save_path, config)

    for idx, dmt in enumerate(dmt_data):
        pdmt = preprocess_img(dmt)
        img = torch.from_numpy(pdmt).permute(2, 0, 1).float().unsqueeze(0)
        with torch.no_grad():
            hm, wh, offset = model(img)
            hm = hm.to(device)
            wh = wh.to(device)
            offset = offset.to(device)
            top_conf, top_boxes = get_res(hm, wh, offset, confidence=0.35)
            if top_boxes is not None:
                print("FRB detected!")
                title = f"{file_basename}_{idx}"
                plot_candidate(dmt, detect_dir, config, title=title + "_candidate")
                plot_img(pdmt, detect_dir, config, title=title + "_detect")
