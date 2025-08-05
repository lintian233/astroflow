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
from ..plotter import plot_dmtime
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
DM_TOLERANCE = 15
TOA_TOLERANCE = 0.2

# File extensions
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


def gen_frb_dmt(
    source: SpectrumBase,
    dm: float,
    toa: float,
    dm_low: float,
    dm_high: float,
    dm_step: float,
    freq_start: float,
    freq_end: float,
    t_sample: float,
    time_downsample: int = 1,
) -> Tuple[Optional[Any], Optional[float]]:
    """Generate FRB Dynamic-DM vs Time plot for given parameters.

    Args:
        source: Input spectrum data
        dm: Dispersion measure value
        toa: Time of arrival
        dm_low: Lower bound of DM search range
        dm_high: Upper bound of DM search range
        dm_step: Step size for DM search
        freq_start: Start frequency
        freq_end: End frequency
        t_sample: Time sampling interval
        time_downsample: Time downsampling factor

    Returns:
        Tuple containing the matching DMT object and reference TOA,
        or (None, None) if no matching DMT is found
    """
    dm_time_series = dedisperse_spec(
        source,
        dm_low,
        dm_high,
        freq_start,
        freq_end,
        dm_step,
        time_downsample,
        t_sample,
    )
    reference_toa = get_ref_freq_toa(source.header(), freq_end, toa, dm)

    for dm_time in dm_time_series:
        if dm_time.tstart < reference_toa < dm_time.tend:
            return dm_time, reference_toa

    return None, None


def _create_label_studio_annotation(
    ref_toa: float, 
    dm: float, 
    dmt: Any, 
    img_path: str,
    imgsize: int = DEFAULT_IMAGE_SIZE,
    width: int = DEFAULT_BBOX_WIDTH
) -> Dict[str, Any]:
    """Create Label Studio annotation format."""
    x = round((ref_toa - dmt.tstart) / (dmt.tend - dmt.tstart) * 100, 3) - DEFAULT_BBOX_OFFSET
    y = round((dm - dmt.dm_low) / (dmt.dm_high - dmt.dm_low) * 100, 3) - DEFAULT_BBOX_OFFSET

    label_studio_json = {
        "result": [
            {
                "type": "rectanglelabels",
                "original_width": imgsize,
                "original_height": imgsize,
                "from_name": "label",
                "to_name": "image",
                "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": width,
                    "rotation": 0,
                    "rectanglelabels": ["object"],
                },
            }
        ]
    }
    
    return {
        "data": {
            "image": f"/data/local-files/?d={img_path}",
        },
        "annotations": [label_studio_json],
    }


def _save_label_data(label_path: str, new_task: Dict[str, Any]) -> None:
    """Save label data to file, handling existing data."""
    existing_data = []
    
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        try:
            with open(label_path, "r") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error reading existing label file {label_path}: {e}")
            existing_data = []
    
    existing_data.append(new_task)
    
    try:
        with open(label_path, "w") as f:
            json.dump(existing_data, f, indent=4)
    except Exception as e:
        logger.error(f"Error writing label file {label_path}: {e}")
        raise


def generate_frb_candidate(
    file_path: str,
    dm: float,
    toa: float,
    dm_low: float,
    dm_high: float,
    dm_step: float,
    freq_start: float,
    freq_end: float,
    t_sample: float,
    label_path: str,
    png_path: str,
    time_downsample: int = 1,
) -> None:
    """Generate FRB candidate with DMT plot and label annotation."""
    source = _load_spectrum_data(file_path)

    dmt, ref_toa = gen_frb_dmt(
        source, dm, toa, dm_low, dm_high, dm_step,
        freq_start, freq_end, t_sample, time_downsample,
    )

    if dmt is None or ref_toa is None:
        raise ValueError("No matching DMT found for the given parameters.")

    # Generate plot
    plot_dmtime(dmt, png_path, imgsize=DEFAULT_IMAGE_SIZE)

    # Create annotation
    img_path = f"{png_path}/{dmt.__str__()}.png"
    task = _create_label_studio_annotation(ref_toa, dm, dmt, img_path)
    
    # Save label data
    _save_label_data(label_path, task)


def generate_frb_dataset(
    dataset_path: str,
    candidate_path: str,
    dm_low: float,
    dm_high: float,
    dm_step: float,
    freq_start: float,
    freq_end: float,
    t_sample: float,
    label_path: str,
    png_path: str,
    time_downsample: int = 1,
) -> None:
    """Generate FRB dataset from candidate file."""
    if not candidate_path.endswith(".csv"):
        raise ValueError("Candidate path must be a CSV file.")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    try:
        candidate_table = pd.read_csv(candidate_path)
    except Exception as e:
        raise ValueError(f"Error reading candidate CSV file: {e}")

    all_files = os.listdir(dataset_path)
    for file in tqdm(all_files):
        if not any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            continue
            
        file_path = os.path.join(dataset_path, file)
        base_name = os.path.basename(file_path)
        candidate = candidate_table[candidate_table["file"] == base_name]
        
        if candidate.empty:
            continue
            
        try:
            dm = candidate["dms"].values[0]
            toa = candidate["toa"].values[0]
            
            generate_frb_candidate(
                file_path, dm, toa, dm_low, dm_high, dm_step,
                freq_start, freq_end, t_sample, label_path, png_path, time_downsample,
            )
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")


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

    # Setup output directories
    file_basename = os.path.basename(file).split(".")[0]
    detect_dir = os.path.join(output_dir, "detect", file_basename).lower()
    candidate_detect_dir = os.path.join(output_dir, "candidate", file_basename).lower()
    background_dir = os.path.join(output_dir, "background").lower()
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
        6,
    )
    
    if task_config.modelname == CENTERNET:
        detector = CenterNetFrbDetector(
            task_config.dm_limt, task_config.preprocess, task_config.confidence
        )
    elif task_config.modelname == YOLOV11N:
        detector = Yolo11nFrbDetector(
            task_config.dm_limt, task_config.preprocess, task_config.confidence
        )
    else:
        logger.warning(f"Unknown model name {task_config.modelname}, using CenterNet")
        detector = CenterNetFrbDetector(
            task_config.dm_limt, task_config.preprocess, task_config.confidence
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
                
                cached_dir = os.path.join(output_dir, "cached").lower()
                file_dir = os.path.join(cached_dir, base_dir, file_basename).lower()
                
                
                if os.path.exists(file_dir):
                    logger.info(f"Skipping already processed file: {file_basename}")
                    # 检查 candidate_detect_dir 目录下是否有文件，如果没有则 detection_flag = 0
                    candidate_detect_dir = os.path.join(output_dir, "candidate", file_basename).lower()
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
        
        logger.info(f"current candidates: {current_candidates}/{total_candidates}")

    logger.info(f"Total candidates found: {current_candidates}/{total_candidates}")
    logger.info(f"Missed candidates: {len(missed_candiates)}")
    logger.info(f"Missed candidate files: {', '.join(missed_candiates)}")
