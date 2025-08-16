import os
import time

import numpy as np
import pandas as pd
import tqdm

from .dedispered import dedispered_fil, dedisperse_spec, dedisperse_spec_with_dm
from .frbdetector import (
    BinaryChecker,
    CenterNetFrbDetector,
    ResNetBinaryChecker,
    Yolo11nFrbDetector,
)

from .dataset.generate import get_freq_end_toa
from .logger import logger  # type: ignore

from .io.filterbank import Filterbank, FilterbankPy
from .io.psrfits import PsrFits
from .plotter import PlotterManager, plot_dmtime
from .utils import Config, SingleDmConfig
from .config.taskconfig import TaskConfig, CENTERNET, YOLOV11N, DETECTNET, COMBINENET


def single_pulsar_search_with_dm_dir(
    task_config: TaskConfig,
):
    if files_dir[-1] == "/":
        files_dir = files_dir[:-1]
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]

    confidence = task_config.confidence
    files_dir = task_config.input
    output_dir = task_config.output

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
    # save_path = os.path.join(output_dir, file_basename).lower()

    os.makedirs(detect_dir, exist_ok=True)
    # os.makedirs(save_path, exist_ok=True)

    for idx, data in enumerate(dmtimes):
        candidate = detector.detect(data)
        for i, candinfo in enumerate(candidate):
            dm = candinfo[0]
            toa = candinfo[1]
            ref_toa = get_freq_end_toa(
            origin_data.header(),candinfo[3], toa, dm
            )
            candinfo.append(idx)
            candinfo.append(ref_toa)
            logger.info(
                f"Found FRB in {file_basename} at DM: {candinfo[0]} at time: {candinfo[1]}"
            )
            plotter.plot_candidate(data, candinfo, detect_dir, file)
    del origin_data
    return candidate


def muti_pulsar_search(
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

    taskconfig = TaskConfig()
    base_name = os.path.basename(file).split(".")[0]
    mask_file_dir = taskconfig.maskdir
    mask_file = f"{mask_file_dir}/{base_name}_your_rfi_mask.bad_chans"
    
    if not os.path.exists(mask_file):
        mask_file = taskconfig.maskfile

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
    # save_path = os.path.join(output_dir, file_basename).lower()

    os.makedirs(detect_dir, exist_ok=True)
    # os.makedirs(save_path, exist_ok=True)

    candidate = detector.mutidetect(dmtimes)
    for i, candinfo in enumerate(candidate):
        dm = candinfo[0]
        toa = candinfo[1]
        ref_toa = get_freq_end_toa(
            origin_data.header(),candinfo[3], toa, dm
        )
        candinfo.append(ref_toa)
        plotter.plot_candidate(dmtimes[candinfo[4]], candinfo, detect_dir, file)
    del origin_data
    return candidate


def single_pulsar_search_dir(task_config: TaskConfig) -> None:

    files_dir = task_config.input
    output_dir = task_config.output
    confidence = task_config.confidence

    if files_dir[-1] == "/":
        files_dir = files_dir[:-1]
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]

    all_files = os.listdir(files_dir)
    mutidetect = False
    plotter = PlotterManager(
        task_config.dmtconfig,
        task_config.specconfig,
        6,
    )
    if task_config.modelname == CENTERNET:
        frb_detector = CenterNetFrbDetector(
            task_config.dm_limt, task_config.preprocess, confidence
        )
    elif task_config.modelname == YOLOV11N:
        frb_detector = Yolo11nFrbDetector(
            task_config.dm_limt, task_config.preprocess, confidence
        )
        mutidetect = True
    else:
        frb_detector = CenterNetFrbDetector(
            task_config.dm_limt, task_config.preprocess, confidence
        )

    dms = task_config.dmrange
    freq_range = task_config.freqrange
    tsamples = task_config.tsample

    for file in tqdm.tqdm(all_files):
        if not file.endswith(".fil") and not file.endswith(".fits"):
            continue

        file_path = os.path.join(files_dir, file)
        print(f"Processing {file_path}")

        for dm_item in dms:
            dm_low = dm_item["dm_low"]
            dm_high = dm_item["dm_high"]
            dm_step = dm_item["dm_step"]
            dm_name = dm_item["name"]

            for freq_item in freq_range:
                freq_start = freq_item["freq_start"]
                freq_end = freq_item["freq_end"]
                freq_name = freq_item["name"]

                for tsample_item in tsamples:
                    t_sample = tsample_item["t"]
                    t_name = tsample_item["name"]

                    config = Config(
                        dm_low=dm_low,
                        dm_high=dm_high,
                        dm_step=dm_step,
                        freq_start=freq_start,
                        freq_end=freq_end,
                        t_sample=t_sample,
                        confidence=confidence,
                        time_downsample=task_config.timedownfactor,
                    )
                    cacheddir = os.path.join(output_dir, "cached").lower()
                    base_dir = os.path.basename(files_dir)
                    base_dir += f"-{config.dm_low}DM-{config.dm_high}DM"
                    base_dir += f"-{config.freq_start}MHz-{config.freq_end}MHz"
                    base_dir += f"-{config.dm_step}DM-{config.t_sample}s"

                    os.makedirs(cacheddir, exist_ok=True)
                    file_dir = os.path.join(
                        cacheddir, base_dir, file.split(".")[0]
                    ).lower()

                    print(f"checking {file_dir}")
                    if os.path.exists(file_dir):
                        print(f"跳过已处理文件: {file} (输出目录 {file_dir} 已存在)")
                        continue

                    logger.info(
                        f"Processing {file} with DM: {dm_name}, Freq: {freq_name}, TSample: {t_name}"
                    )
                    logger.info(
                        f"DM Range: {dm_low}-{dm_high}, Freq Range: {freq_start}-{freq_end}, TSample: {t_sample}"
                    )

                    try:
                        if mutidetect:
                            muti_pulsar_search(
                                file_path,
                                output_dir,
                                config,
                                frb_detector,
                                plotter,
                            )
                        else:
                            single_pulsar_search(
                                file_path,
                                output_dir,
                                config,
                                frb_detector,
                                plotter,
                            )
                        os.makedirs(file_dir, exist_ok=True)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    plotter.close()


def single_pulsar_search_file(task_config: TaskConfig) -> None:

    confidence = task_config.confidence
    file = task_config.input
    file = os.path.abspath(file)
    print(f"Processing file: {file}")
    mutildetect = False
    if task_config.modelname == CENTERNET:
        frb_detector = CenterNetFrbDetector(
            task_config.dm_limt, task_config.preprocess, confidence
        )
    elif task_config.modelname == YOLOV11N:
        frb_detector = Yolo11nFrbDetector(
            task_config.dm_limt, task_config.preprocess, confidence
        )
        mutildetect = True
    else:
        frb_detector = CenterNetFrbDetector(
            task_config.dm_limt, task_config.preprocess, confidence
        )

    plotter = PlotterManager(task_config.dmtconfig, task_config.specconfig, 6)
    dms = task_config.dmrange
    dm_range = task_config.dmrange
    freq_range = task_config.freqrange
    tsmaple = task_config.tsample
    for dm_item in dms:
        dm_low = dm_item["dm_low"]
        dm_high = dm_item["dm_high"]
        dm_step = dm_item["dm_step"]
        dm_name = dm_item["name"]

        for freq_item in freq_range:
            freq_start = freq_item["freq_start"]
            freq_end = freq_item["freq_end"]
            freq_name = freq_item["name"]

            for tsample in tsmaple:
                t_sample = tsample["t"]
                t_name = tsample["name"]

                config = Config(
                    dm_low=dm_low,
                    dm_high=dm_high,
                    dm_step=dm_step,
                    freq_start=freq_start,
                    freq_end=freq_end,
                    t_sample=t_sample,
                    confidence=confidence,
                    time_downsample=task_config.timedownfactor,
                )

                logger.info(
                    f"Processing {file} with DM: {dm_name}, Freq: {freq_name}, TSample: {t_name}"
                )
                logger.info
                (
                    f"DM Range: {dm_low}-{dm_high}, Freq Range: {freq_start}-{freq_end}, TSample: {t_sample}"
                )
                if mutildetect:
                    muti_pulsar_search(
                        file,
                        task_config.output,
                        config,
                        frb_detector,
                        plotter,
                    )
                else:
                    single_pulsar_search(
                        file,
                        task_config.output,
                        config,
                        frb_detector,
                        plotter,
                    )

    plotter.close()

