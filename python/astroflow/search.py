import os
import time
from typing import Union, Optional, List, Tuple, Set

import numpy as np
import pandas as pd
from tqdm import tqdm


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
    """
    Validate file path and format.
    
    Parameters
    ----------
    file_path : str
        Path to the file to be validated
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path
    ValueError
        If the file format is not supported
        
    Examples
    --------
    >>> _validate_file_path('/path/to/data.fil')
    >>> _validate_file_path('/path/to/data.fits')
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1]
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(SUPPORTED_EXTENSIONS.keys())}")


def _load_spectrum_data(file_path: str) -> SpectrumBase:
    """
    Load spectrum data based on file extension.
    
    Parameters
    ----------
    file_path : str
        Path to the spectrum data file. Supported formats are .fil and .fits
        
    Returns
    -------
    SpectrumBase
        Loaded spectrum data object, either Filterbank or PsrFits instance
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is not supported
        
    Examples
    --------
    >>> data = _load_spectrum_data('/path/to/observation.fil')
    >>> header = data.header()
    >>> print(f"Observation duration: {header.ndata * header.tsamp} seconds")
    """
    _validate_file_path(file_path)
    
    ext = os.path.splitext(file_path)[1]
    data_class = SUPPORTED_EXTENSIONS[ext]
    return data_class(file_path)


def _create_detector_and_plotter(task_config: TaskConfig) -> Tuple[Union[CenterNetFrbDetector, Yolo11nFrbDetector], PlotterManager, bool]:
    """
    Create detector and plotter instances based on configuration.
    
    Parameters
    ----------
    task_config : TaskConfig
        Configuration object containing model parameters, detection settings,
        and plotting configuration
        
    Returns
    -------
    detector : Union[CenterNetFrbDetector, Yolo11nFrbDetector]
        FRB detector instance configured according to the model specified
        in task_config.modelname
    plotter : PlotterManager
        Plotter manager instance for generating detection plots
    mutidetect : bool
        Whether the detector supports multi-detection mode
        
    Notes
    -----
    Supported model names:
    - CENTERNET: Uses CenterNetFrbDetector
    - YOLOV11N: Uses Yolo11nFrbDetector with multi-detection enabled
    
    If an unknown model name is provided, defaults to CenterNetFrbDetector
    with a warning.
    
    Examples
    --------
    >>> task_config = TaskConfig()
    >>> task_config.modelname = YOLOV11N
    >>> detector, plotter, multi = _create_detector_and_plotter(task_config)
    >>> print(f"Multi-detection enabled: {multi}")
    """
    plotter = PlotterManager(
        task_config.dmtconfig,
        task_config.specconfig,
        task_config.plotworker,
    )
    
    mutidetect = False
    if task_config.modelname == CENTERNET:
        detector = CenterNetFrbDetector(
            task_config.dm_limt, None, task_config.confidence
        )
    elif task_config.modelname == YOLOV11N:
        detector = Yolo11nFrbDetector(
            task_config.dm_limt, None, task_config.confidence
        )
        mutidetect = True
    else:
        logger.warning(f"Unknown model name {task_config.modelname}, using CenterNet")
        detector = CenterNetFrbDetector(
            task_config.dm_limt, None, task_config.confidence
        )
    
    return detector, plotter, mutidetect


def _normalize_path(path: str) -> str:
    """
    Normalize directory path by removing trailing slash.
    
    Parameters
    ----------
    path : str
        Directory path that may contain trailing slashes
        
    Returns
    -------
    str
        Normalized path with trailing slashes removed
        
    Examples
    --------
    >>> _normalize_path('/home/user/data/')
    '/home/user/data'
    >>> _normalize_path('/home/user/data')
    '/home/user/data'
    """
    return path.rstrip("/")


def _get_cached_dir_path(output_dir: str, files_dir: str, config: Config) -> str:
    """
    Generate cached directory path for configuration.
    
    Parameters
    ----------
    output_dir : str
        Base output directory where cached results will be stored
    files_dir : str
        Directory containing the input files
    config : Config
        Configuration object containing DM range, frequency range,
        and time sampling parameters
        
    Returns
    -------
    str
        Full path to the cached directory with configuration-specific naming
        
    Notes
    -----
    The cached directory name is constructed from:
    - Base name of the input files directory
    - DM range (dm_low, dm_high)
    - Frequency range (freq_start, freq_end)
    - DM step size
    - Time sample duration
    
    The resulting path structure is:
    {output_dir}/cached/{base_dir}-{dm_low}DM-{dm_high}DM-{freq_start}MHz-{freq_end}MHz-{dm_step}DM-{t_sample}s
    
    Examples
    --------
    >>> config = Config(dm_low=0, dm_high=100, freq_start=1200, freq_end=1500, 
    ...                 dm_step=0.1, t_sample=0.001)
    >>> path = _get_cached_dir_path('/output', '/data/observations', config)
    >>> print(path)
    /output/cached/observations-0dm-100dm-1200mhz-1500mhz-0.1dm-0.001s
    """
    base_dir = os.path.basename(files_dir)
    base_dir += f"-{config.dm_low}DM-{config.dm_high}DM"
    base_dir += f"-{config.freq_start}MHz-{config.freq_end}MHz" 
    base_dir += f"-{config.dm_step}DM-{config.t_sample}s"
    
    cached_dir = os.path.join(output_dir, "cached")
    return os.path.join(cached_dir, base_dir)


def single_pulsar_search_with_dm(
    file: str,
    output_dir: str,
    config: SingleDmConfig,
    checker: BinaryChecker,
    plotter: PlotterManager,
) -> None:
    """
    Perform single pulsar search with specific DM.
    
    Parameters
    ----------
    file : str
        Path to the input spectrum file (.fil or .fits format)
    output_dir : str
        Directory where detection results and plots will be saved
    config : SingleDmConfig
        Configuration containing specific DM value, frequency range,
        and time sampling parameters
    checker : BinaryChecker
        Binary checker instance for candidate detection
    plotter : PlotterManager
        Plotter manager for generating detection plots
        
    Notes
    -----
    This function performs the following steps:
    1. Load spectrum data from the input file
    2. Dedisperse the spectrum using the specified DM value
    3. Check for pulsar candidates using the binary checker
    4. Generate plots for detected candidates
    
    Output files are saved in {output_dir}/detect/ with naming convention:
    {basename}-dm-{dm}-tstart-{tstart}-tend-{tend}
    
    Examples
    --------
    >>> config = SingleDmConfig(dm=50.0, freq_start=1200, freq_end=1500, t_sample=0.001)
    >>> checker = ResNetBinaryChecker()
    >>> plotter = PlotterManager(...)
    >>> single_pulsar_search_with_dm('obs.fil', '/output', config, checker, plotter)
    """
    raise NotImplementedError("This Method Not imp yet")

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
        return


def single_pulsar_search(
    file: str,
    output_dir: str,
    config: Config,
    detector: Union[CenterNetFrbDetector, Yolo11nFrbDetector],
    plotter: PlotterManager,
) -> List:
    """
    Perform single pulsar search on a file.
    
    Parameters
    ----------
    file : str
        Path to the input spectrum file (.fil or .fits format)
    output_dir : str
        Directory where detection results and plots will be saved
    config : Config
        Configuration containing DM range, frequency range, time sampling,
        and other search parameters
    detector : Union[CenterNetFrbDetector, Yolo11nFrbDetector]
        FRB detector instance for candidate detection
    plotter : PlotterManager
        Plotter manager for generating detection plots
        
    Returns
    -------
    List
        List of detected candidates with their properties
        
    Notes
    -----
    This function performs the following steps:
    1. Load spectrum data from the input file
    2. Generate dedispersed time series across the DM range
    3. Apply FRB detection to each DM trial
    4. Generate plots for all detected candidates
    5. Clean up memory by deleting the original data
    
    The search covers DM values from config.dm_low to config.dm_high
    with step size config.dm_step. Each detection is plotted and saved
    in the detect subdirectory.
    
    Examples
    --------
    >>> config = Config(dm_low=0, dm_high=100, dm_step=0.1, 
    ...                 freq_start=1200, freq_end=1500, t_sample=0.001)
    >>> detector = CenterNetFrbDetector(...)
    >>> plotter = PlotterManager(...)
    >>> candidates = single_pulsar_search('obs.fil', '/output', config, detector, plotter)
    >>> print(f"Found {len(candidates)} candidates")
    """
    origin_data = _load_spectrum_data(file)

    maskfile = TaskConfig().maskfile

    dmtimes = dedisperse_spec(
        origin_data,
        config.dm_low,
        config.dm_high,
        config.freq_start,
        config.freq_end,
        config.dm_step,
        config.time_downsample,
        config.t_sample,
        maskfile=maskfile
    )

    detect_dir = os.path.join(output_dir, "detect")
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
    """
    Perform multi pulsar search on a file.
    
    Parameters
    ----------
    file : str
        Path to the input spectrum file (.fil or .fits format)
    output_dir : str
        Directory where detection results and plots will be saved
    config : Config
        Configuration containing DM range, frequency range, time sampling,
        and other search parameters
    detector : Union[CenterNetFrbDetector, Yolo11nFrbDetector]
        FRB detector instance that supports multi-detection mode
    plotter : PlotterManager
        Plotter manager for generating detection plots
        
    Returns
    -------
    List
        List of detected candidates with their properties. Each candidate
        contains position and confidence information.
        
    Notes
    -----
    This function differs from single_pulsar_search by:
    1. Using multi-detection mode on the detector
    2. Attempting to load RFI mask files for better detection
    3. Processing all DM trials simultaneously for efficiency
    
    RFI mask file search order:
    1. {maskdir}/{basename}_your_rfi_mask.bad_chans
    2. Default mask file from task configuration
    
    The multi-detection approach can be more efficient for detecting
    multiple candidates simultaneously across different DM trials.
    
    Examples
    --------
    >>> config = Config(dm_low=0, dm_high=100, dm_step=0.1)
    >>> detector = Yolo11nFrbDetector(...)  # Supports multi-detection
    >>> candidates = muti_pulsar_search('obs.fil', '/output', config, detector, plotter)
    >>> for i, cand in enumerate(candidates):
    ...     print(f"Candidate {i}: DM={cand[4]}, confidence={cand[5]}")
    """
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
        maskfile=mask_file
    )

    detect_dir = os.path.join(output_dir, "detect")
    file_basename = os.path.basename(file).split(".")[0]
    
    os.makedirs(detect_dir, exist_ok=True)
    # start_time = time.time()
    candidates = detector.mutidetect(dmtimes)
    # end_time = time.time()
    # print(f"[TIMER] Detect: {end_time - start_time:.2f} s")
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
    """
    Process a single file with all parameter combinations for search.
    
    Parameters
    ----------
    file_path : str
        Path to the spectrum file to be processed
    task_config : TaskConfig
        Task configuration containing parameter ranges for DM, frequency,
        and time sampling
    detector : Union[CenterNetFrbDetector, Yolo11nFrbDetector]
        FRB detector instance for candidate detection
    plotter : PlotterManager
        Plotter manager for generating detection plots
    output_dir : str
        Base output directory for results
    mutidetect : bool
        Whether to use multi-detection mode
        
    Notes
    -----
    This function iterates through all combinations of:
    - DM ranges from task_config.dmrange
    - Frequency ranges from task_config.freqrange  
    - Time sampling values from task_config.tsample
    
    For each combination, it:
    1. Creates a Config object with the specific parameters
    2. Checks if results already exist in cache
    3. Performs the appropriate search (single or multi)
    4. Creates a cache marker directory to avoid reprocessing
    
    Caching prevents redundant processing of the same file with
    identical parameters.
    
    Examples
    --------
    >>> task_config = TaskConfig()
    >>> task_config.dmrange = [{"dm_low": 0, "dm_high": 50, "dm_step": 0.1}]
    >>> detector = CenterNetFrbDetector(...)
    >>> _process_single_file_search('obs.fil', task_config, detector, 
    ...                           plotter, '/output', False)
    """
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
                file_dir = os.path.join(cached_dir_path, file_basename)
                
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
    """
    Perform pulsar search on all files in a directory.
    
    Parameters
    ----------
    task_config : TaskConfig
        Task configuration containing:
        - input: Directory path containing spectrum files
        - output: Output directory for results
        - Model and detection parameters
        - DM, frequency, and time sampling ranges
        
    Raises
    ------
    FileNotFoundError
        If the input directory does not exist
        
    Notes
    -----
    This function processes all supported files (.fil, .fits) in the
    input directory. For each file, it:
    
    1. Validates the input directory exists
    2. Finds all supported spectrum files
    3. Initializes detector and plotter based on configuration
    4. Processes each file with all parameter combinations
    5. Handles errors gracefully and continues processing
    
    Progress is displayed using tqdm progress bars.
    The plotter is properly closed after processing to free resources.
    
    Supported file formats:
    - .fil (Filterbank format)
    - .fits (PSRFITS format)
    
    Examples
    --------
    >>> task_config = TaskConfig()
    >>> task_config.input = '/data/observations'
    >>> task_config.output = '/results'
    >>> task_config.modelname = CENTERNET
    >>> single_pulsar_search_dir(task_config)
    """
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
        for file in tqdm(all_files):
            file_path = os.path.join(files_dir, file)
            logger.info(f"Processing {file_path}")

            _process_single_file_search(
                file_path, task_config, detector, plotter, output_dir, mutidetect
            )
    finally:
        plotter.close()


def single_pulsar_search_file(task_config: TaskConfig) -> None:
    """
    Perform pulsar search on a single file.
    
    Parameters
    ----------
    task_config : TaskConfig
        Task configuration containing:
        - input: Path to single spectrum file
        - output: Output directory for results  
        - Model and detection parameters
        - DM, frequency, and time sampling ranges
        
    Raises
    ------
    FileNotFoundError
        If the input file does not exist
        
    Notes
    -----
    This function processes a single spectrum file with all parameter
    combinations specified in the task configuration. For each combination:
    
    1. Creates a Config object with specific parameters
    2. Logs processing information
    3. Performs appropriate search (single or multi-detection)
    4. Handles errors gracefully and continues with next combination
    
    The function iterates through all combinations of:
    - DM ranges (dm_low, dm_high, dm_step)
    - Frequency ranges (freq_start, freq_end)
    - Time sampling values (t_sample)
    
    Each combination is logged with descriptive names and parameter values
    for tracking progress and debugging.
    
    Examples
    --------
    >>> task_config = TaskConfig()
    >>> task_config.input = '/data/observation.fil'
    >>> task_config.output = '/results'
    >>> task_config.modelname = YOLOV11N
    >>> task_config.dmrange = [{"name": "Low DM", "dm_low": 0, "dm_high": 50, "dm_step": 0.1}]
    >>> single_pulsar_search_file(task_config)
    """
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


def _get_supported_files(directory: str) -> Set[str]:
    """
    Get all supported files (.fil, .fits) in a directory.
    
    Parameters
    ----------
    directory : str
        Directory path to scan for supported files
        
    Returns
    -------
    Set[str]
        Set of full file paths for all supported files found
        
    Examples
    --------
    >>> files = _get_supported_files('/data/observations')
    >>> print(f"Found {len(files)} supported files")
    """
    if not os.path.exists(directory):
        return set()
    
    supported_files = set()
    try:
        for filename in os.listdir(directory):
            if any(filename.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                full_path = os.path.join(directory, filename)
                if os.path.isfile(full_path):
                    supported_files.add(full_path)
    except PermissionError:
        logger.warning(f"Permission denied accessing directory: {directory}")
    except Exception as e:
        logger.error(f"Error scanning directory {directory}: {e}")
    
    return supported_files


def _process_new_file(
    file_path: str,
    task_config: TaskConfig,
    detector: Union[CenterNetFrbDetector, Yolo11nFrbDetector],
    plotter: PlotterManager,
    mutidetect: bool
) -> None:
    """
    Process a newly detected file with all parameter combinations.
    
    Parameters
    ----------
    file_path : str
        Path to the new file to be processed
    task_config : TaskConfig
        Task configuration containing processing parameters
    detector : Union[CenterNetFrbDetector, Yolo11nFrbDetector]
        FRB detector instance for candidate detection
    plotter : PlotterManager
        Plotter manager for generating detection plots
    mutidetect : bool
        Whether to use multi-detection mode
        
    Notes
    -----
    This function processes a single new file with all parameter combinations
    from the task configuration. It handles errors gracefully to avoid
    interrupting the monitoring process.
    """
    logger.info(f"Processing new file: {file_path}")
    
    try:
        _process_single_file_search(
            file_path, task_config, detector, plotter, 
            task_config.output, mutidetect
        )
        logger.info(f"Successfully processed: {file_path}")
    except Exception as e:
        logger.error(f"Error processing new file {file_path}: {e}")


def monitor_directory_for_pulsar_search(
    task_config: TaskConfig, 
    check_interval: float = 3.0,
    stop_file: Optional[str] = None
) -> None:
    """
    Monitor a directory for new pulsar data files and process them automatically.
    
    Parameters
    ----------
    task_config : TaskConfig
        Task configuration containing:
        - input: Directory path to monitor for new files
        - output: Output directory for results
        - Model and detection parameters
        - DM, frequency, and time sampling ranges
    check_interval : float, optional
        Time interval in seconds between directory checks (default: 3.0)
    stop_file : str, optional
        Path to a stop file. If this file exists, monitoring will stop.
        If None, monitoring continues indefinitely until interrupted.
        
    Raises
    ------
    FileNotFoundError
        If the input directory does not exist
    KeyboardInterrupt
        When monitoring is interrupted by user (Ctrl+C)
        
    Notes
    -----
    This function continuously monitors the specified directory for new
    .fil and .fits files. When new files are detected:
    
    1. Each new file is immediately processed with all parameter combinations
    2. Files are tracked to avoid reprocessing
    3. Errors in processing individual files don't stop monitoring
    4. Progress and status are logged for tracking
    
    The monitoring can be stopped by:
    - Keyboard interrupt (Ctrl+C)
    - Creating a stop file (if stop_file parameter is provided)
    
    Memory usage is managed by maintaining a set of processed files
    rather than keeping file contents in memory.
    
    Examples
    --------
    >>> task_config = TaskConfig()
    >>> task_config.input = '/data/incoming'
    >>> task_config.output = '/results'
    >>> # Monitor every 3 seconds, stop when /tmp/stop_monitoring exists
    >>> monitor_directory_for_pulsar_search(task_config, 3.0, '/tmp/stop_monitoring')
    
    >>> # Monitor indefinitely until Ctrl+C
    >>> monitor_directory_for_pulsar_search(task_config)
    """
    monitor_dir = _normalize_path(task_config.input)
    output_dir = _normalize_path(task_config.output)
    
    if not os.path.exists(monitor_dir):
        raise FileNotFoundError(f"Monitor directory not found: {monitor_dir}")
    
    logger.info(f"Starting directory monitoring for: {monitor_dir}")
    logger.info(f"Check interval: {check_interval} seconds")
    logger.info(f"Output directory: {output_dir}")
    if stop_file:
        logger.info(f"Stop file: {stop_file}")
    logger.info("Press Ctrl+C to stop monitoring")
    
    # Initialize detector and plotter
    detector, plotter, mutidetect = _create_detector_and_plotter(task_config)
    
    # Track processed files to avoid reprocessing
    processed_files: Set[str] = set()
    
    # Initial scan to populate processed files set
    logger.info("Performing initial directory scan...")
    processed_files = _get_supported_files(monitor_dir)
    logger.info(f"Found {len(processed_files)} existing files (will be ignored)")
    
    try:
        while True:
            # Check for stop file if specified
            if stop_file and os.path.exists(stop_file):
                logger.info(f"Stop file detected: {stop_file}")
                break
            
            # Get current files in directory
            current_files = _get_supported_files(monitor_dir)
            
            # Find new files
            new_files = current_files - processed_files
            
            if new_files:
                logger.info(f"Detected {len(new_files)} new file(s)")
                
                # Process each new file
                for file_path in tqdm(sorted(new_files)):
                    _process_new_file(
                        file_path, task_config, detector, plotter, mutidetect
                    )
                    
                    # Add to processed set immediately after processing
                    processed_files.add(file_path)
            else:
                logger.debug(f"No new files detected. Waiting {check_interval} seconds...")
            
            # Wait before next check
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    except Exception as e:
        logger.error(f"Monitoring error: {e}")
    finally:
        logger.info("Stopping directory monitoring...")
        plotter.close()
        logger.info("Directory monitoring stopped")

