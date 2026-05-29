from __future__ import annotations

import multiprocessing
from collections.abc import Mapping, Sequence

from .dmtime import DmTime
from .plotting.analysis import calculate_frb_snr
from .plotting.pipeline import close_plot_sessions as _close_plot_sessions
from .plotting.pipeline import pack_background as _pack_background
from .plotting.pipeline import pack_candidate as _pack_candidate
from .plotting.pipeline import plot_candidate as _plot_candidate
from .plotting.pipeline import plot_candidates_for_path as _plot_candidates_for_path
from .plotting.pipeline import save_candidate_metrics_for_path as _save_candidate_metrics_for_path
from .plotting.pipeline import save_fast_candidate_info_for_path as _save_fast_candidate_info_for_path
from .plotting.types import (
    CandidateInfo,
    DmPlotConfig,
    SpecPlotConfig,
    ensure_candidate_info,
    ensure_dmt_config,
    ensure_spec_config,
)


def _init_plot_worker(taskconfig_snapshot):
    from .config.taskconfig import TaskConfig

    TaskConfig.initialize_from_snapshot(taskconfig_snapshot)


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
        from .config.taskconfig import TaskConfig

        taskconfig = TaskConfig()
        self.onlycand = taskconfig.onlycand
        self.fastcand = taskconfig.fastcand
        self.max_worker = 1 if self.onlycand or self.fastcand else max_worker
        ctx = multiprocessing.get_context("spawn")
        self.pool = ctx.Pool(
            self.max_worker,
            initializer=_init_plot_worker,
            initargs=(taskconfig.snapshot(),),
        )
        self.dmtconfig = ensure_dmt_config(dmtconfig)
        self.specconfig = ensure_spec_config(specconfig)
        self.speconfig = self.specconfig

    def pack_background(self, dmt: DmTime, candinfo, save_path, file_path):
        if self.onlycand or self.fastcand:
            return
        candinfo = _to_pickle_safe(candinfo)
        self.pool.apply_async(_pack_background, args=(dmt, candinfo, save_path, file_path))

    def pack_candidate(self, dmt: DmTime, candinfo, save_path, file_path):
        if self.onlycand or self.fastcand:
            return
        candinfo = _to_pickle_safe(candinfo)
        self.pool.apply_async(_pack_candidate, args=(dmt, candinfo, save_path, file_path))

    def plot_candidate(self, dmt: DmTime, candinfo, save_path, file_path):
        candinfo = _to_pickle_safe(candinfo)
        if self.fastcand:
            self.pool.apply_async(
                _save_fast_candidate_info_for_path,
                args=(file_path, [(None, candinfo, save_path)]),
            )
            return
        if self.onlycand:
            self.pool.apply_async(
                _save_candidate_metrics_for_path,
                args=(file_path, [(dmt, candinfo, save_path)], self.dmtconfig, self.specconfig),
            )
            return
        self.pool.apply_async(
            _plot_candidate,
            args=(dmt, candinfo, save_path, file_path, self.dmtconfig, self.specconfig),
        )

    def plot_candidates_for_file(self, file_path, candidates, dpi=100):
        candidates = _to_pickle_safe(candidates)
        if self.fastcand:
            self.pool.apply_async(
                _save_fast_candidate_info_for_path,
                args=(file_path, _strip_fastcand_candidates(candidates)),
            )
            return
        if self.onlycand:
            self.pool.apply_async(
                _save_candidate_metrics_for_path,
                args=(file_path, candidates, self.dmtconfig, self.specconfig),
            )
            return
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


def save_candidate_metrics_for_path(file_path, candidates, dmtconfig, specconfig):
    dmtconfig = ensure_dmt_config(dmtconfig)
    specconfig = ensure_spec_config(specconfig)
    return _save_candidate_metrics_for_path(file_path, candidates, dmtconfig, specconfig)


def save_fast_candidate_info_for_path(file_path, candidates):
    return _save_fast_candidate_info_for_path(file_path, candidates)


def plot_candidates_for_file(file_path, candidates, dmtconfig, specconfig, dpi=150):
    return plot_candidates_for_path(file_path, candidates, dmtconfig, specconfig, dpi)


def close_plot_sessions():
    return _close_plot_sessions()


def _strip_fastcand_candidates(candidates):
    return [(None, _to_pickle_safe(candinfo), save_path) for _dmt, candinfo, save_path in candidates]


def _to_pickle_safe(value):
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().numpy().tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _to_pickle_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_pickle_safe(item) for item in value)
    if isinstance(value, list):
        return [_to_pickle_safe(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(_to_pickle_safe(item) for item in value)
    return value


__all__ = [
    "CandidateInfo",
    "DmPlotConfig",
    "SpecPlotConfig",
    "PlotterManager",
    "calculate_frb_snr",
    "close_plot_sessions",
    "error_tracer",
    "pack_background",
    "pack_candidate",
    "plot_candidates_for_file",
    "plot_candidates_for_path",
    "plot_candidate",
    "save_candidate_metrics_for_path",
    "save_fast_candidate_info_for_path",
]
