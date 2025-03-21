import os
import multiprocessing
from astroflow import dedispered_fil
from astroflow.utils import draw_dm_times
import time
from astroflow.utils import Config, timeit
from center_net import detect_frb


import torch
from centernet_model import centernet
from centernet_utils import get_res

timeit_astroflow = timeit(dedispered_fil)


def dedispered_dir(dir, config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = "resnet50"  # 'resnet50'
    model = centernet(model_name=base_model)
    model.load_state_dict(torch.load("cent_{}.pth".format(base_model)))
    model.eval()
    all_files = os.listdir(dir)
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
        detect_frb(file_path, config, base_dir, model, device)


def search_frb_with_dir():
    dir_path = "/data/QL/predata/7399_FRB20250316Spec_25-03-20_19-11-29"
    dm_low = 0
    dm_high = 300
    freq_start = 1300
    freq_end = 1500
    dm_step = 1
    time_downsample = 1
    t_sample = 0.5
    config = Config(
        dm_low, dm_high, freq_start, freq_end, dm_step, time_downsample, t_sample
    )
    dedispered_dir(dir_path, config)


def search_frb_with_single():
    file_path = "/home/miaocc/work/FRBsearch/sim_FRB/b0329_40us_nsrt.fil"
    dm_low = 0
    dm_high = 800
    freq_start = 1130
    freq_end = 1448
    dm_step = 1
    time_downsample = 1
    t_sample = 0.5
    config = Config(
        dm_low, dm_high, freq_start, freq_end, dm_step, time_downsample, t_sample
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = "resnet50"  # 'resnet50'
    model = centernet(model_name=base_model)
    model.load_state_dict(torch.load("cent_{}.pth".format(base_model)))
    model.eval()

    detect_frb(file_path, config, "ql", model, device)


def draw_candidate_frb(file, config: Config):
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
    draw_dm_times(data, "ql/" + file.split(".")[0].split("/")[-1])


search_frb_with_dir()
