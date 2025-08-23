import os
import json
from turtle import back
from tqdm import tqdm
import pandas as pd
from typing import Tuple, Optional, Union, Dict, Any, List

from ..dedispered import dedisperse_spec
from ..io.data import Header, SpectrumBase
from ..io.filterbank import Filterbank
from ..io.psrfits import PsrFits
from ..config.taskconfig import TaskConfig, YOLOV11N, CENTERNET
from ..logger import logger
from ..utils import Config, SingleDmConfig
from ..plotter import PlotterManager
from ..frbdetector import CenterNetFrbDetector, Yolo11nFrbDetector

# Constants
DISPERSION_CONSTANT = 4148.808
DEFAULT_IMAGE_SIZE = 512
DEFAULT_BBOX_WIDTH = 20
DEFAULT_BBOX_OFFSET = 10
DM_TOLERANCE = 20
TOA_TOLERANCE = 0.3

# File extensions
SUPPORTED_EXTENSIONS = {'.fil': Filterbank, '.fits': PsrFits}


def _validate_file_path(file_path: str) -> None:
    """Validate file path and format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1]
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(SUPPORTED_EXTENSIONS.keys())}")


def _load_spectrum_data(file_path: str) -> SpectrumBase:
    """Load spectrum data based on file extension."""
    _validate_file_path(file_path)
    
    ext = os.path.splitext(file_path)[1]
    data_class = SUPPORTED_EXTENSIONS[ext]
    return data_class(file_path)


def get_ref_freq_toa(header: Header, ref_freq: float, freq_end_toa: float, dm: float) -> float:
    """Calculate time of arrival at reference frequency."""
    fch1 = header.fch1
    foff = header.foff
    nchan = header.nchans
    freq_end = fch1 + foff * (nchan - 1)
    time_latency = DISPERSION_CONSTANT * dm * (1 / (ref_freq**2) - 1 / (freq_end**2))
    return freq_end_toa + time_latency


def get_freq_end_toa(header: Header, ref_freq: float, ref_freq_toa: float, dm: float) -> float:
    """Convert TOA from reference frequency to header's freq_end frequency.
    
    Args:
        header: Header object containing frequency information
        ref_freq: Reference frequency where ref_freq_toa is measured
        ref_freq_toa: Time of arrival at reference frequency
        dm: Dispersion measure value
        
    Returns:
        Time of arrival at header's freq_end frequency
    """
    fch1 = header.fch1
    foff = header.foff
    nchan = header.nchans
    freq_end = fch1 + foff * (nchan - 1)
    time_latency = DISPERSION_CONSTANT * dm * (1 / (ref_freq**2) - 1 / (freq_end**2))
    return ref_freq_toa - time_latency


def _check_candidate_match(
    detected_dm: float, 
    detected_toa: float, 
    origin_dm: float, 
    origin_toa: float, 
    ref_toa: float
) -> bool:
    """Check if detected candidate matches original candidate within tolerance."""
    dm_match = abs(detected_dm - origin_dm) < DM_TOLERANCE
    toa_match = abs(origin_toa - ref_toa) < TOA_TOLERANCE
    return dm_match and toa_match

def _check_dm_match(
    detected_dm: float, 
    origin_dm: float
) -> bool:
    """Check if detected DM matches original DM within tolerance."""
    return abs(detected_dm - origin_dm) < DM_TOLERANCE


def muti_pulsar_search_detect(
    file: str,
    output_dir: str,
    config: Config,
    detector: Union[CenterNetFrbDetector, Yolo11nFrbDetector],
    plotter: PlotterManager,
    frbcandidate: Optional[pd.DataFrame] = None,
) -> int:
    """Perform multi-pulsar search and detection."""
    origin_data = _load_spectrum_data(file)


    taskconfig = TaskConfig()
    basename = os.path.basename(file).split(".")[0]
    mask_file_dir = taskconfig.maskdir
    maskfile = f"{mask_file_dir}/{basename}_your_rfi_mask.bad_chans"
    
    if not os.path.exists(maskfile):
        maskfile = taskconfig.maskfile

    dmtimes = dedisperse_spec(
        origin_data,
        config.dm_low,
        config.dm_high,
        config.freq_start,
        config.freq_end,
        config.dm_step,
        config.time_downsample,
        config.t_sample,
        maskfile=maskfile,
    )

    # Setup output directories
    file_basename = os.path.basename(file).split(".")[0]
    detect_dir = os.path.join(output_dir, "detect", file_basename)
    candidate_detect_dir = os.path.join(output_dir, "candidate", file_basename)
    background_dir = os.path.join(output_dir, "background")
    os.makedirs(candidate_detect_dir, exist_ok=True)
    os.makedirs(detect_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)

    # Get detections
    candidates = detector.mutidetect(dmtimes)
    
    if frbcandidate is None or frbcandidate.empty:
        logger.warning("No FRB candidate data provided")
        return 0
        
    origin_toa = frbcandidate["toa"].values[0]
    origin_dm = frbcandidate["dms"].values[0]
    
    detection_flag = 0
    
    for candinfo in candidates: #type: ignore
        dm, toa = candinfo[0], candinfo[1]
        ref_toa = get_freq_end_toa(
            origin_data.header(), 
            ref_freq=config.freq_end, 
            ref_freq_toa=toa, 
            dm=origin_dm
        )
        candinfo.append(ref_toa)
        
        if _check_candidate_match(dm, toa, origin_dm, origin_toa, ref_toa):
            plotter.plot_candidate(dmtimes[candinfo[4]], candinfo, candidate_detect_dir, file)
            plotter.pack_candidate(dmtimes[candinfo[4]], candinfo, output_dir, file)
            detection_flag = 1
        elif _check_dm_match(dm, origin_dm):
            plotter.plot_candidate(dmtimes[candinfo[4]], candinfo, detect_dir, file)
        else:
            plotter.plot_candidate(dmtimes[candinfo[4]], candinfo, background_dir, file)
            plotter.pack_background(dmtimes[candinfo[4]], candinfo, background_dir, file)

    return detection_flag


def _create_detector_and_plotter(task_config: TaskConfig) -> Tuple[Union[CenterNetFrbDetector, Yolo11nFrbDetector], PlotterManager]:
    """Create detector and plotter instances based on configuration."""
    plotter = PlotterManager(
        task_config.dmtconfig,
        task_config.specconfig,
        task_config.plotworker,
    )
    
    if task_config.modelname == CENTERNET:
        detector = CenterNetFrbDetector(
            task_config.dm_limt, None, task_config.confidence
        )
    elif task_config.modelname == YOLOV11N:
        detector = Yolo11nFrbDetector(
            task_config.dm_limt, None, task_config.confidence
        )
    else:
        logger.warning(f"Unknown model name {task_config.modelname}, using CenterNet")
        detector = CenterNetFrbDetector(
            task_config.dm_limt, None, task_config.confidence
        )
    
    return detector, plotter


def _process_single_file(
    file_path: str,
    candidate: pd.DataFrame,
    task_config: TaskConfig,
    detector: Union[CenterNetFrbDetector, Yolo11nFrbDetector],
    plotter: PlotterManager,
    output_dir: str
) -> bool:
    """Process a single file with all parameter combinations."""
    file_detected = False
    
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
                base_dir = os.path.basename(task_config.input)
                base_dir += f"-{config.dm_low}DM-{config.dm_high}DM"
                base_dir += f"-{config.freq_start}MHz-{config.freq_end}MHz"
                base_dir += f"-{config.dm_step}DM-{config.t_sample}s"
                
                cached_dir = os.path.join(output_dir, "cached")
                file_dir = os.path.join(cached_dir, base_dir, file_basename)
                
                
                if os.path.exists(file_dir):
                    logger.info(f"Skipping already processed file: {file_basename}")
                    # 检查 candidate_detect_dir 目录下是否有文件，如果没有则 detection_flag = 0
                    candidate_detect_dir = os.path.join(output_dir, "candidate", file_basename)
                    if any(os.scandir(candidate_detect_dir)):
                        file_detected = True
                    continue


                try:
                    detection_flag = muti_pulsar_search_detect(
                        file_path, output_dir, config, detector, plotter, candidate
                    )
                    if detection_flag == 1:
                        file_detected = True
                    os.makedirs(file_dir, exist_ok=True)
                except Exception as e:
                    logger.error(f"Error processing {file_path} with config: {e}")
    
    return file_detected


def count_frb_dataset(
    dataset_path: str,
    candidate_path: str,
    task_config: TaskConfig,
) -> None:
    """Count FRB candidates in dataset and generate detection statistics."""
    if not candidate_path.endswith(".csv"):
        raise ValueError("Candidate path must be a CSV file.")

    try:
        candidate_table = pd.read_csv(candidate_path)
    except Exception as e:
        raise ValueError(f"Error reading candidate CSV file: {e}")

    # Normalize directory paths
    files_dir = task_config.input.rstrip("/")
    output_dir = task_config.output.rstrip("/")

    if not os.path.exists(files_dir):
        raise FileNotFoundError(f"Input directory not found: {files_dir}")

    # Initialize detector and plotter
    detector, plotter = _create_detector_and_plotter(task_config)

    # Process files
    all_files = sorted([f for f in os.listdir(files_dir) 
                       if any(f.endswith(ext) for ext in SUPPORTED_EXTENSIONS)])
    
    total_candidates = len(candidate_table)
    current_candidates = 0
    missed_candiates = []
    for i, file in enumerate(tqdm(all_files)):
        file_path = os.path.join(files_dir, file)
        base_name = os.path.basename(file_path)
        candidate = candidate_table[candidate_table["file"] == base_name]
    
        if candidate.empty:
            logger.warning(f"No candidate data found for file: {base_name}")
            
            continue
        logger.info(f"Processing {file_path}")
        
        try:
            file_detected = _process_single_file(
                file_path, candidate, task_config, detector, plotter, output_dir
            )
            
            if file_detected:
                current_candidates += 1
            else:
                missed_candiates.append(file)
                logger.error(f"No candidates found for {file} with the given parameters.")
                
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
        
        # CC Recall MISS
        logger.info(f"CC: {current_candidates}/{i + 1} R: {current_candidates / (i + 1) * 100:.2f}% MISS: {len(missed_candiates)}")
    
    plotter.close()
    logger.info(f"Total candidates found: {current_candidates}/{total_candidates}")
    logger.info(f"Missed candidates: {len(missed_candiates)}")
    print(f"Missed candidate files: {'\n '.join(missed_candiates)}")
