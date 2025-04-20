import os
import time

import numpy as np
import pandas as pd
import tqdm

from .dedispered import dedispered_fil, dedisperse_spec, dedisperse_spec_with_dm
from .frbdetector import BinaryChecker, CenterNetFrbDetector, ResNetBinaryChecker
from .io.filterbank import Filterbank, FilterbankPy
from .io.psrfits import PsrFits
from .plotter import PlotterManager
from .utils import Config, SingleDmConfig


def single_pulsar_search_with_dm_dir(
    files_dir: str, output_dir: str, config: SingleDmConfig, confidence=0.99
):
    if files_dir[-1] == "/":
        files_dir = files_dir[:-1]
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]

    all_files = os.listdir(files_dir)
    base_dir = os.path.basename(files_dir)
    base_dir += f"-{config.dm}DM-{config.freq_start}MHz-{config.freq_end}MHz"

    frbchecker = ResNetBinaryChecker(confidence=confidence)
    plotter = PlotterManager()
    for file in tqdm.tqdm(all_files):
        if not file.endswith(".fil") and not file.endswith(".fits"):
            continue

        file_path = os.path.join(files_dir, file)
        print(f"Processing {file_path}")

        cached = "cached"
        file_dir = os.path.join(
            output_dir, cached, base_dir, file.split(".")[0]
        ).lower()

        print(f"checking {file_dir}")
        if os.path.exists(file_dir):
            print(f"跳过已处理文件: {file} (输出目录 {file_dir} 已存在)")
            continue

        try:
            single_pulsar_search_with_dm(
                file_path,
                os.path.join(output_dir, base_dir),
                config,
                frbchecker,
                plotter,
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        os.makedirs(file_dir, exist_ok=True)
    plotter.close()


def single_pulsar_search_with_dm_file(
    file: str,
    output_dir: str,
    config: SingleDmConfig,
    confidence: float = 0.99,
):
    checker = ResNetBinaryChecker(confidence=confidence)
    plotter = PlotterManager()
    single_pulsar_search_with_dm(
        file,
        output_dir,
        config,
        checker,
        plotter,
    )
    plotter.close()


def single_pulsar_search_with_dm(
    file: str,
    output_dir: str,
    config: SingleDmConfig,
    checker: BinaryChecker,
    plotter: PlotterManager,
) -> None:
    if file.endswith(".fil"):
        origin_data = Filterbank(file)
    elif file.endswith(".fits"):
        origin_data = PsrFits(file)
    else:
        raise ValueError("Unknown file type")

    os.makedirs(output_dir, exist_ok=True)
    detect_dir = os.path.join(output_dir, "detect")
    os.makedirs(detect_dir, exist_ok=True)

    header = origin_data.header()
    tsamp = header.tsamp
    ndata = header.ndata
    tstart = 0
    tend = ndata * tsamp
    spec = dedisperse_spec_with_dm(
        origin_data,
        tstart,
        tend,
        config.dm,
        config.freq_start,
        config.freq_end,
    )
    candidates = checker.check(spec, config.t_sample)
    spec_datas = spec.clip(t_sample=config.t_sample)
    spec_datas = spec_datas[candidates]
    for idx, spec_data in enumerate(spec_datas):
        tstart = spec.tstart + candidates[idx] * config.t_sample
        tend = tstart + config.t_sample
        freq_start = spec.freq_start
        freq_end = spec.freq_end
        tstart = np.round(tstart, 3)
        tend = np.round(tend, 3)
        basename = os.path.basename(file).split(".")[0]
        title = f"{basename}-dm-{config.dm}-tstart-{tstart}-tend-{tend}"
        plotter.plot_spec(
            spec_data,
            title,
            [tstart, tend, freq_start, freq_end],
            detect_dir,
        )


def single_pulsar_search(
    file: str,
    output_dir: str,
    config: Config,
    detector: CenterNetFrbDetector,
    plotter: PlotterManager,
) -> None:

    origin_data = None
    if file.endswith(".fil"):
        origin_data = Filterbank(file)
    elif file.endswith(".fits"):
        origin_data = PsrFits(file)
    else:
        raise ValueError("Unknown file type")

    dmtimes = dedisperse_spec(
        origin_data,
        config.dm_low,
        config.dm_high,
        config.freq_start,
        config.freq_end,
        config.dm_step,
        config.time_downsample,
        config.t_sample,
    )
    detect_dir = os.path.join(output_dir, "detect").lower()
    file_basename = os.path.basename(file).split(".")[0]
    save_path = os.path.join(output_dir, file_basename).lower()

    os.makedirs(detect_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    for idx, data in enumerate(dmtimes):
        candidate = detector.detect(data)
        # plotter.plot_dmtime(data, save_path)
        for i, candinfo in enumerate(candidate):
            print(
                f"Found FRB in {file_basename} at DM: {candinfo[0]} at time: {candinfo[1]}"
            )
            plotter.plot_candidate(data, candinfo, detect_dir, file)
            # plotter.plot_spectrogram(file, candinfo, detect_dir)

    del origin_data


def single_pulsar_search_dir(files_dir: str, output_dir: str, config: Config) -> None:

    if files_dir[-1] == "/":
        files_dir = files_dir[:-1]
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]

    all_files = os.listdir(files_dir)
    base_dir = os.path.basename(files_dir)
    base_dir += f"-{config.dm_low}DM-{config.dm_high}DM"
    base_dir += f"-{config.freq_start}MHz-{config.freq_end}MHz"
    base_dir += f"-{config.dm_step}DM-{config.t_sample}s"

    plotter = PlotterManager()

    frb_detector = CenterNetFrbDetector(confidence=config.confidence)
    plotter = PlotterManager(8)
    for file in tqdm.tqdm(all_files):
        if not file.endswith(".fil") and not file.endswith(".fits"):
            continue

        file_dir = os.path.join(output_dir, base_dir, file.split(".")[0]).lower()

        print(f"checking {file_dir}")
        if os.path.exists(file_dir):
            print(f"跳过已处理文件: {file} (输出目录 {file_dir} 已存在)")
            continue

        file_path = os.path.join(files_dir, file)
        print(f"Processing {file_path}")
        try:
            single_pulsar_search(
                file_path,
                os.path.join(output_dir, base_dir),
                config,
                frb_detector,
                plotter,
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plotter.close()


def single_pulsar_search_file(file: str, output_dir: str, config: Config) -> None:
    plotter = PlotterManager(3)
    frb_detector = CenterNetFrbDetector(confidence=config.confidence)
    single_pulsar_search(file, output_dir, config, frb_detector, plotter)
    plotter.close()
