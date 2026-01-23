from __future__ import annotations

import fcntl
from typing import Mapping

from ..io.filterbank import Filterbank
from ..io.psrfits import PsrFits


def load_data_file(file_path: str):
    """Load filterbank or psrfits data file."""
    if file_path.endswith(".fil"):
        return Filterbank(file_path)
    if file_path.endswith(".fits"):
        return PsrFits(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def save_candidate_info(file_path: str, cand_info: Mapping[str, object]) -> None:
    """Atomically appends candidate information to a CSV-like file."""
    header = "file,mjd,dms,toa,toa_ref_freq_end,snr,pulse_width_ms,freq_start,freq_end,file_path,plot_path"

    values = [
        cand_info.get("file", ""),
        cand_info.get("mjd", ""),
        cand_info.get("dms", ""),
        cand_info.get("toa", ""),
        cand_info.get("toa_ref_freq_end", ""),
        cand_info.get("snr", ""),
        cand_info.get("pulse_width_ms", ""),
        cand_info.get("freq_start", ""),
        cand_info.get("freq_end", ""),
        cand_info.get("file_path", ""),
        cand_info.get("plot_path", ""),
    ]
    line = ",".join(map(str, values))

    with open(file_path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(header + "\n")
            handle.write(line + "\n")
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
