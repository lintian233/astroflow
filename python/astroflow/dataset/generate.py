import os
import json
from tqdm import tqdm
import pandas as pd

from ..dedispered import dedisperse_spec
from ..io.data import Header, SpectrumBase
from typing import Tuple, Optional
from ..io.filterbank import Filterbank
from ..io.psrfits import PsrFits
from ..plotter import plot_dmtime
from ..config.taskconfig import TaskConfig,YOLOV11N,CENTERNET


from ..logger import logger  # type: ignore
from ..utils import Config, SingleDmConfig
from ..plotter import PlotterManager
from ..frbdetector import CenterNetFrbDetector, Yolo11nFrbDetector




def get_ref_freq_toa(header: Header, ref_freq: float, freq_end_toa: float, dm: float):

    fch1 = header.fch1
    foff = header.foff
    nchan = header.nchans
    freq_end = fch1 + foff * (nchan - 1)
    time_latency = 4148.808 * dm * (1 / (ref_freq**2) - 1 / (freq_end**2))
    ref_freq_toa = freq_end_toa + time_latency

    return ref_freq_toa


def get_freq_end_toa(header: Header, ref_freq: float, ref_freq_toa: float, dm: float):
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
    time_latency = 4148.808 * dm * (1 / (ref_freq**2) - 1 / (freq_end**2))
    freq_end_toa = ref_freq_toa - time_latency

    return freq_end_toa


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
):
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
):
    if file_path.endswith(".fil"):
        source = Filterbank(file_path)
    elif file_path.endswith(".fits"):
        source = PsrFits(file_path)
    else:
        raise ValueError("Unsupported file format. Only .fil and .fits are supported.")

    dmt, ref_toa = gen_frb_dmt(
        source,
        dm,
        toa,
        dm_low,
        dm_high,
        dm_step,
        freq_start,
        freq_end,
        t_sample,
        time_downsample,
    )

    if dmt is None:
        raise ValueError("No matching DMT found for the given parameters.")

    imgsize = 512

    plot_dmtime(dmt, png_path, imgsize=imgsize)

    x = round((ref_toa - dmt.tstart) / (dmt.tend - dmt.tstart) * 100, 3) - 10
    y = round((dm - dmt.dm_low) / (dmt.dm_high - dmt.dm_low) * 100, 3) - 10

    img_path = f"{png_path}/{dmt.__str__()}.png"
    task = []
    width = 20
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
    task.append(
        {
            "data": {
                "image": f"/data/local-files/?d={img_path}",
            },
            "annotations": [label_studio_json],
        }
    )

    # 追加写入 label_path
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        try:
            with open(label_path, "r") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                existing.extend(task)
            else:
                existing = task
        except Exception:
            existing = task
    else:
        existing = task
    with open(label_path, "w") as f:
        json.dump(existing, f, indent=4)


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
):
    if not candidate_path.endswith("csv"):
        raise ValueError("Candidate path must be a CSV file.")

    candidate_table = pd.read_csv(candidate_path)

    all_files = os.listdir(dataset_path)
    for file in tqdm(all_files):
        file_path = os.path.join(dataset_path, file)
        base_name = os.path.basename(file_path)
        candidate = candidate_table[candidate_table["file"] == base_name]
        if candidate.empty:
            continue
        dm = candidate["dms"].values[0]
        toa = candidate["toa"].values[0]
        try:
            generate_frb_candidate(
                file_path,
                dm,
                toa,
                dm_low,
                dm_high,
                dm_step,
                freq_start,
                freq_end,
                t_sample,
                label_path,
                png_path,
                time_downsample,
            )
        except Exception as e:
            print(f"Error processing {file}: {e}")


def muti_pulsar_search_detect(
    file: str,
    output_dir: str,
    config: Config,
    detector: CenterNetFrbDetector,
    plotter: PlotterManager,
    frbcandidate: Optional[pd.DataFrame] = None,
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

    
    file_basename = os.path.basename(file).split(".")[0]
    detect_dir = os.path.join(
        output_dir, "detect", file_basename
    ).lower()
    candidate_detect_dir = os.path.join(
        output_dir, "candidate", file_basename
    ).lower()

    # save_path = os.path.join(output_dir, file_basename).lower()
    os.makedirs(candidate_detect_dir, exist_ok=True)
    os.makedirs(detect_dir, exist_ok=True)
    

    candidate = detector.mutidetect(dmtimes)
    origin_toa = frbcandidate["toa"].values[0]
    origin_dm = frbcandidate["dms"].values[0]
    print(f"origin_toa: {origin_toa}, origin_dm: {origin_dm}")
    flag = 0
    for i, candinfo in enumerate(candidate):
        dm = candinfo[0]
        toa = candinfo[1]
        freq_start = config.freq_start
        freq_end = config.freq_end
        ref_toa = get_freq_end_toa(
            origin_data.header(), ref_freq=freq_end, ref_freq_toa=toa, dm=origin_dm
        )
        if abs(dm - origin_dm) < 15 and abs(origin_toa - ref_toa) < 0.2:
            candinfo.append(ref_toa)
            plotter.plot_candidate(dmtimes[candinfo[4]], candinfo, candidate_detect_dir, file)
            flag = 1
        else:
            candinfo.append(ref_toa)  # add ref_toa to candidate info
            plotter.plot_candidate(
                dmtimes[candinfo[4]], candinfo, detect_dir, file
            )
    return flag
            
    
def count_frb_dataset(
    dataset_path: str,
    candidate_path: str,
    task_config: TaskConfig,
):
    """
    对于给定的FRB数据集，统计满足条件的候选事件数量。
    对于每一个dataset中的文件，搜索候选体，得到的candiateinfo，去匹配candidate_path中的候选体。
    若dm，toa 都满足，则认为是一个正确的候选体。
    然后把不同的候选体输出到不同的目录中，并生成统计结果。
    """
    if not candidate_path.endswith("csv"):
        raise ValueError("Candidate path must be a CSV file.")

    candidate_table = pd.read_csv(candidate_path)

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
    total_candidates = len(candidate_table)
    current_candidates = 0
    # sort
    all_files = sorted(all_files)
    for i, file in enumerate(tqdm(all_files)):
        cand_flag = False
        if not file.endswith(".fil") and not file.endswith(".fits"):
            continue

        file_path = os.path.join(files_dir, file)
        print(f"Processing {file_path}")
        base_name = os.path.basename(file_path)
        candidate = candidate_table[candidate_table["file"] == base_name]
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
                        flag = 1
                        continue

                    try:
                        flag = muti_pulsar_search_detect(
                            file_path,
                            output_dir,
                            config,
                            frb_detector,
                            plotter,
                            candidate,
                        )
                        if flag == 1:
                            cand_flag = True
                        os.makedirs(file_dir, exist_ok=True)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        if cand_flag:
            current_candidates += 1
        else:
            logger.error(
                f"No candidates found for {file} with the given parameters."
            )
        logger.info(f"processed {file}")
        logger.info(
            f"Processed {file} - Current Candidates: {current_candidates}/{i+1}"
        )

    print(f"Total candidates found: {current_candidates}/{total_candidates}")
        
