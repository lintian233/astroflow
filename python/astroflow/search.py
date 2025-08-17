import os
import time
from typing import Union, Optional, List, Tuple

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
from .io.data import SpectrumBase
from .plotter import PlotterManager
from .utils import Config, SingleDmConfig
from .config.taskconfig import TaskConfig, CENTERNET, YOLOV11N, DETECTNET, COMBINENET

# Constants
SUPPORTED_EXTENSIONS = {'.fil': Filterbank, '.fits': PsrFits}


def _validate_file_path(file_path: str) -> None:
    """Validate file path and format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(SUPPORTED_EXTENSIONS.keys())}")


def _load_spectrum_data(file_path: str) -> SpectrumBase:
    """Load spectrum data based on file extension."""
    _validate_file_path(file_path)
    
    ext = os.path.splitext(file_path)[1].lower()
    data_class = SUPPORTED_EXTENSIONS[ext]
    return data_class(file_path)


def _create_detector_and_plotter(task_config: TaskConfig) -> Tuple[Union[CenterNetFrbDetector, Yolo11nFrbDetector], PlotterManager, bool]:
    """Create detector and plotter instances based on configuration."""
    plotter = PlotterManager(
        task_config.dmtconfig,
        task_config.specconfig,
        6,
    )
    
    mutidetect = False
    if task_config.modelname == CENTERNET:
        detector = CenterNetFrbDetector(
            task_config.dm_limt, task_config.preprocess, task_config.confidence
        )
    elif task_config.modelname == YOLOV11N:
        detector = Yolo11nFrbDetector(
            task_config.dm_limt, task_config.preprocess, task_config.confidence
        )
        mutidetect = True
    else:
        logger.warning(f"Unknown model name {task_config.modelname}, using CenterNet")
        detector = CenterNetFrbDetector(
            task_config.dm_limt, task_config.preprocess, task_config.confidence
        )
    
    return detector, plotter, mutidetect


def _normalize_path(path: str) -> str:
    """Normalize directory path by removing trailing slash."""
    return path.rstrip("/")


def _get_cached_dir_path(output_dir: str, files_dir: str, config: Config) -> str:
    """Generate cached directory path for configuration."""
    base_dir = os.path.basename(files_dir)
    base_dir += f"-{config.dm_low}DM-{config.dm_high}DM"
    base_dir += f"-{config.freq_start}MHz-{config.freq_end}MHz" 
    base_dir += f"-{config.dm_step}DM-{config.t_sample}s"
    
    cached_dir = os.path.join(output_dir, "cached").lower()
    return os.path.join(cached_dir, base_dir)


def single_pulsar_search_with_dm(
    file: str,
    output_dir: str,
    config: SingleDmConfig,
    checker: BinaryChecker,
    plotter: PlotterManager,
) -> None:
    """Perform single pulsar search with specific DM."""
    origin_data = _load_spectrum_data(file)

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
    detector: Union[CenterNetFrbDetector, Yolo11nFrbDetector],
    plotter: PlotterManager,
) -> List:
    """Perform single pulsar search on a file."""
    origin_data = _load_spectrum_data(file)

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

    os.makedirs(detect_dir, exist_ok=True)

    candidates = []
    for idx, data in enumerate(dmtimes):
        candidate = detector.detect(data)
        for i, candinfo in enumerate(candidate):
            plotter.plot_candidate(data, candinfo, detect_dir, file)
            candidates.extend(candidate)
    
    del origin_data
    return candidates


def muti_pulsar_search(
    file: str,
    output_dir: str,
    config: Config,
    detector: Union[CenterNetFrbDetector, Yolo11nFrbDetector],
    plotter: PlotterManager,
) -> List:
    """Perform multi pulsar search on a file."""
    origin_data = _load_spectrum_data(file)

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

    os.makedirs(detect_dir, exist_ok=True)

    candidates = detector.mutidetect(dmtimes)
    if candidates is None:
        candidates = []
        
    for i, candinfo in enumerate(candidates):
        plotter.plot_candidate(dmtimes[candinfo[4]], candinfo, detect_dir, file)
    
    del origin_data
    return candidates

    
def _process_single_file_search(
    file_path: str,
    task_config: TaskConfig,
    detector: Union[CenterNetFrbDetector, Yolo11nFrbDetector],
    plotter: PlotterManager,
    output_dir: str,
    mutidetect: bool
) -> None:
    """Process a single file with all parameter combinations for search."""
    for dm_item in task_config.dmrange:
        for freq_item in task_config.freqrange:
            for tsample_item in task_config.tsample:
                config = Config(
                    dm_low=dm_item["dm_low"],
                    dm_high=dm_item["dm_high"],
                    dm_step=dm_item["dm_step"],
                    freq_start=freq_item["freq_start"],
                    freq_end=freq_item["freq_end"],
                    t_sample=tsample_item["t"],
                    confidence=task_config.confidence,
                    time_downsample=task_config.timedownfactor,
                )
                
                # Check if already processed
                file_basename = os.path.basename(file_path).split(".")[0]
                cached_dir_path = _get_cached_dir_path(output_dir, task_config.input, config)
                file_dir = os.path.join(cached_dir_path, file_basename).lower()
                
                print(f"checking {file_dir}")
                if os.path.exists(file_dir):
                    continue

                try:
                    if mutidetect:
                        muti_pulsar_search(
                            file_path,
                            output_dir,
                            config,
                            detector,
                            plotter,
                        )
                    else:
                        single_pulsar_search(
                            file_path,
                            output_dir,
                            config,
                            detector,
                            plotter,
                        )
                    os.makedirs(file_dir, exist_ok=True)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")


def single_pulsar_search_dir(task_config: TaskConfig) -> None:
    """Perform pulsar search on all files in a directory."""
    files_dir = _normalize_path(task_config.input)
    output_dir = _normalize_path(task_config.output)

    if not os.path.exists(files_dir):
        raise FileNotFoundError(f"Input directory not found: {files_dir}")

    # Get all supported files
    all_files = sorted([f for f in os.listdir(files_dir) 
                       if any(f.endswith(ext) for ext in SUPPORTED_EXTENSIONS)])
    
    if not all_files:
        logger.warning(f"No supported files found in {files_dir}")
        return

    # Initialize detector and plotter
    detector, plotter, mutidetect = _create_detector_and_plotter(task_config)

    try:
        for file in tqdm.tqdm(all_files):
            file_path = os.path.join(files_dir, file)
            logger.info(f"Processing {file_path}")

            _process_single_file_search(
                file_path, task_config, detector, plotter, output_dir, mutidetect
            )
    finally:
        plotter.close()


def single_pulsar_search_file(task_config: TaskConfig) -> None:
    """Perform pulsar search on a single file."""
    file_path = os.path.abspath(task_config.input)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Processing file: {file_path}")
    
    # Initialize detector and plotter
    detector, plotter, mutidetect = _create_detector_and_plotter(task_config)

    try:
        for dm_item in task_config.dmrange:
            for freq_item in task_config.freqrange:
                for tsample_item in task_config.tsample:
                    config = Config(
                        dm_low=dm_item["dm_low"],
                        dm_high=dm_item["dm_high"],
                        dm_step=dm_item["dm_step"],
                        freq_start=freq_item["freq_start"],
                        freq_end=freq_item["freq_end"],
                        t_sample=tsample_item["t"],
                        confidence=task_config.confidence,
                        time_downsample=task_config.timedownfactor,
                    )

                    logger.info(
                        f"Processing {file_path} with DM: {dm_item['name']}, "
                        f"Freq: {freq_item['name']}, TSample: {tsample_item['name']}"
                    )
                    logger.info(
                        f"DM Range: {config.dm_low}-{config.dm_high}, "
                        f"Freq Range: {config.freq_start}-{config.freq_end}, "
                        f"TSample: {config.t_sample}"
                    )
                    
                    try:
                        if mutidetect:
                            muti_pulsar_search(
                                file_path,
                                task_config.output,
                                config,
                                detector,
                                plotter,
                            )
                        else:
                            single_pulsar_search(
                                file_path,
                                task_config.output,
                                config,
                                detector,
                                plotter,
                            )
                    except Exception as e:
                        logger.error(f"Error processing {file_path} with config: {e}")
    finally:
        plotter.close()

