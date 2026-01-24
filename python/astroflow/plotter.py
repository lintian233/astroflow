from __future__ import annotations

import multiprocessing

from .dmtime import DmTime
from .plotting.analysis import calculate_frb_snr
from .plotting.pipeline import pack_background as _pack_background
from .plotting.pipeline import pack_candidate as _pack_candidate
from .plotting.pipeline import plot_candidate as _plot_candidate
from .plotting.pipeline import plot_candidates_for_path as _plot_candidates_for_path
from .plotting.types import (
    CandidateInfo,
    DmPlotConfig,
    SpecPlotConfig,
    ensure_candidate_info,
    ensure_dmt_config,
    ensure_spec_config,
)


def error_tracer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(f"Error in {func.__name__}: {exc}")
            raise

    return wrapper


class PlotterManager:
    def __init__(self, dmtconfig=None, specconfig=None, max_worker=8):
        self.max_worker = max_worker
        self.pool = multiprocessing.Pool(self.max_worker)
        self.dmtconfig = ensure_dmt_config(dmtconfig)
        self.specconfig = ensure_spec_config(specconfig)
        self.speconfig = self.specconfig

    def pack_background(self, dmt: DmTime, candinfo, save_path, file_path):
        self.pool.apply_async(_pack_background, args=(dmt, candinfo, save_path, file_path))

    def pack_candidate(self, dmt: DmTime, candinfo, save_path, file_path):
        self.pool.apply_async(_pack_candidate, args=(dmt, candinfo, save_path, file_path))

    def plot_candidate(self, dmt: DmTime, candinfo, save_path, file_path):
        self.pool.apply_async(
            _plot_candidate,
            args=(dmt, candinfo, save_path, file_path, self.dmtconfig, self.specconfig),
        )

    def plot_candidates_for_file(self, file_path, candidates, dpi=150):
        self.pool.apply_async(
            _plot_candidates_for_path,
            args=(file_path, candidates, self.dmtconfig, self.specconfig, dpi),
        )

    def close(self):
        self.pool.close()
        self.pool.join()


def pack_candidate(dmt, candinfo, save_path, file_path):
    return _pack_candidate(dmt, ensure_candidate_info(candinfo), save_path, file_path)


def pack_background(dmt, candinfo, save_path, file_path):
    return _pack_background(dmt, ensure_candidate_info(candinfo), save_path, file_path)


def plot_candidate(dmt, candinfo, save_path, file_path, dmtconfig, specconfig, dpi=150):
    dmtconfig = ensure_dmt_config(dmtconfig)
    specconfig = ensure_spec_config(specconfig)
    return _plot_candidate(dmt, candinfo, save_path, file_path, dmtconfig, specconfig, dpi)


def plot_candidates_for_path(file_path, candidates, dmtconfig, specconfig, dpi=150):
    dmtconfig = ensure_dmt_config(dmtconfig)
    specconfig = ensure_spec_config(specconfig)
    return _plot_candidates_for_path(file_path, candidates, dmtconfig, specconfig, dpi)


def plot_candidates_for_file(file_path, candidates, dmtconfig, specconfig, dpi=150):
    return plot_candidates_for_path(file_path, candidates, dmtconfig, specconfig, dpi)


__all__ = [
    "CandidateInfo",
    "DmPlotConfig",
    "SpecPlotConfig",
    "PlotterManager",
    "calculate_frb_snr",
    "error_tracer",
    "pack_background",
    "pack_candidate",
    "plot_candidates_for_file",
    "plot_candidates_for_path",
    "plot_candidate",
]
