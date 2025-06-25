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


def get_ref_freq_toa(header: Header, ref_freq: float, freq_end_toa: float, dm: float):

    fch1 = header.fch1
    foff = header.foff
    nchan = header.nchans
    freq_end = fch1 + foff * (nchan - 1)
    time_latency = 4148.808 * dm * (1 / (ref_freq**2) - 1 / (freq_end**2))
    ref_freq_toa = freq_end_toa + time_latency

    return ref_freq_toa


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
    time_downsample: int = 1
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
        source, dm_low, dm_high, 
        freq_start, freq_end, dm_step, time_downsample, t_sample
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
    time_downsample: int = 1
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
        time_downsample
    )

    if dmt is None:
        raise ValueError("No matching DMT found for the given parameters.")
    
    imgsize = 512

    plot_dmtime(dmt, png_path, imgsize=imgsize)
    
    x = round((ref_toa - dmt.tstart)/(dmt.tend - dmt.tstart) * 100,3) - 10
    y = round((dm - dmt.dm_low)/(dmt.dm_high - dmt.dm_low) * 100,3) - 10

    img_path = f"{png_path}/{dmt.__str__()}.png"
    task = []
    width = 20
    label_studio_json ={
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
        time_downsample: int = 1
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
                time_downsample
            )
        except Exception as e:
            print(f"Error processing {file}: {e}")
