from __future__ import annotations

import gc
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..config.taskconfig import TaskConfig
from ..dedispered import dedisperse_spec_with_dm
from ..utils import get_freq_end_toa
from .analysis import calculate_frb_snr, detrend, downsample_freq_weighted_vec, estimate_peak_width
from .io import load_data_file, save_candidate_info
from .plots import (
    calculate_spectrum_time_window,
    prepare_dm_data,
    setup_detrend_spectrum_plots,
    setup_dm_plots,
    setup_spectrum_plots,
    setup_subband_spectrum_plots,
)
from .types import CandidateInfo, ensure_candidate_info


def pack_candidate(dmt, candinfo, save_path, file_path):
    image_path = os.path.join(save_path, "frb", "images")
    label_path = os.path.join(save_path, "frb", "labels")

    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    cand = ensure_candidate_info(candinfo)
    x, y, w, h = cand.bbox if cand.bbox else (0, 0, 0, 0)
    img = dmt.data

    name = f"dm_{cand.dm}_toa_{cand.ref_toa:.3f}_{dmt.__str__()}.png"
    label_name = f"dm_{cand.dm}_toa_{cand.ref_toa:.3f}_{dmt.__str__()}.txt"

    cv2.imwrite(os.path.join(image_path, name), img)

    with open(os.path.join(label_path, label_name), "w") as handle:
        if cand.bbox is not None:
            handle.write(f"0 {x:.2f} {y:.2f} {w:.2f} {h:.2f} \n")


def pack_background(dmt, candinfo, save_path, file_path):
    image_path = os.path.join(save_path, "bg", "images")
    label_path = os.path.join(save_path, "bg", "labels")

    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    cand = ensure_candidate_info(candinfo)
    img = dmt.data

    name = f"bg_dm_{cand.dm}_toa_{cand.ref_toa:.3f}_{dmt.__str__()}.png"
    label_name = f"bg_dm_{cand.dm}_toa_{cand.ref_toa:.3f}_{dmt.__str__()}.txt"

    cv2.imwrite(os.path.join(image_path, name), img)

    open(os.path.join(label_path, label_name), "w").close()


def plot_candidate(dmt, candinfo, save_path, file_path, dmtconfig, specconfig, dpi=150):
    """
    Plot FRB candidate with DM-Time and spectrum analysis.
    """
    origin_data = load_data_file(file_path)
    try:
        header = origin_data.header()
        taskconfig = TaskConfig()
        maskfile = _resolve_maskfile(taskconfig, file_path)
        _plot_candidate_with_origin(
            origin_data,
            header,
            taskconfig,
            maskfile,
            dmt,
            candinfo,
            save_path,
            file_path,
            dmtconfig,
            specconfig,
            dpi,
        )
    finally:
        _close_origin_data(origin_data)
        gc.collect()


def plot_candidates_for_file(origin_data, file_path, candidates, dmtconfig, specconfig, dpi=150):
    """
    Plot multiple candidates for the same file using a shared IO handle.
    candidates: iterable of (dmt, candinfo, save_path)
    """
    header = origin_data.header()
    taskconfig = TaskConfig()
    maskfile = _resolve_maskfile(taskconfig, file_path)
    for dmt, candinfo, save_path in candidates:
        _plot_candidate_with_origin(
            origin_data,
            header,
            taskconfig,
            maskfile,
            dmt,
            candinfo,
            save_path,
            file_path,
            dmtconfig,
            specconfig,
            dpi,
        )
    gc.collect()


def plot_candidates_for_path(file_path, candidates, dmtconfig, specconfig, dpi=150):
    """
    Plot multiple candidates for the same file by opening the file once in this process.
    """
    origin_data = load_data_file(file_path)
    try:
        plot_candidates_for_file(origin_data, file_path, candidates, dmtconfig, specconfig, dpi)
    finally:
        _close_origin_data(origin_data)
        gc.collect()


def _plot_candidate_with_origin(
    origin_data,
    header,
    taskconfig,
    maskfile,
    dmt,
    candinfo,
    save_path,
    file_path,
    dmtconfig,
    specconfig,
    dpi,
):
    cand = ensure_candidate_info(candinfo)
    try:
        print(
            f"Plot cand: DM={cand.dm}, TOA={cand.toa}, "
            f"Freq={cand.freq_start}-{cand.freq_end} MHz, DMT Index={cand.dmt_idx}"
        )

        fig = plt.figure(figsize=(20, 10), dpi=dpi)
        gs = GridSpec(
            2,
            4,
            figure=fig,
            width_ratios=[3, 1, 3, 1],
            height_ratios=[1, 3],
            wspace=0.04,
            hspace=0.04,
        )

        dm_data, time_axis, dm_axis = prepare_dm_data(dmt)
        dm_vmin, dm_vmax = np.percentile(
            dm_data, [dmtconfig.minpercentile, dmtconfig.maxpercentile]
        )
        setup_dm_plots(fig, gs, dm_data, time_axis, dm_axis, dm_vmin, dm_vmax, cand.dm, cand.toa)

        max_width_samples = _boxcar_max_samples(specconfig, header)

        try:
            mode = _normalize_mode(specconfig.mode)
            ref_toa = get_freq_end_toa(header, cand.freq_end, cand.toa, cand.dm)
            tband = specconfig.tband if specconfig.tband is not None else 0.5
            initial_spec_tstart, initial_spec_tend = calculate_spectrum_time_window(
                cand.toa, 0, header.tsamp, tband
            )

            initial_spectrum = dedisperse_spec_with_dm(
                origin_data,
                initial_spec_tstart,
                initial_spec_tend,
                cand.dm,
                cand.freq_start,
                cand.freq_end,
                maskfile,
            )
            initial_spec_data = initial_spectrum.data

            toa_sample_idx = int((cand.toa - initial_spec_tstart) / header.tsamp)
            toa_sample_idx = max(0, min(toa_sample_idx, initial_spectrum.ntimes - 1))

            initial_time_series = np.sum(initial_spec_data, axis=1, dtype=np.float32)
            search_radius = max_width_samples if max_width_samples is not None else 100
            initial_peak_idx, initial_pulse_width = estimate_peak_width(
                initial_time_series, toa_sample_idx=toa_sample_idx, search_radius=search_radius
            )

            peak_time = initial_spec_tstart + (initial_peak_idx + 0.5) * header.tsamp
            spec_tstart, spec_tend = calculate_spectrum_time_window(
                peak_time, initial_pulse_width, header.tsamp, tband, 35
            )

            spectrum = dedisperse_spec_with_dm(
                origin_data,
                spec_tstart,
                spec_tend,
                cand.dm,
                cand.freq_start,
                cand.freq_end,
                maskfile,
            )

            spec_data = spectrum.data
            spec_time_axis = np.linspace(spec_tstart, spec_tend, spectrum.ntimes)
            spec_freq_axis = np.linspace(cand.freq_start, cand.freq_end, spectrum.nchans)

            subfreq = _resolve_subfreq(specconfig, spec_data.shape[1])
            subband_matrix, _ = downsample_freq_weighted_vec(
                spec_data, spec_freq_axis, subfreq
            )
            snr_input = _snr_input_from_subband(mode, specconfig, subband_matrix)
            snr, pulse_width, peak_idx, _ = calculate_frb_snr(
                snr_input,
                noise_range=None,
                threshold_sigma=5,
                toa_sample_idx=None,
                fitting_window_samples=max_width_samples,
            )

            if snr < taskconfig.snrhold:
                return

            peak_time = spec_tstart + (peak_idx + 0.5) * header.tsamp
            pulse_width_ms = pulse_width * header.tsamp * 1e3 if pulse_width > 0 else -1
            if mode == "subband":
                subband_freq_axis = np.linspace(
                    spec_freq_axis[0], spec_freq_axis[-1], subfreq + 1
                )
                setup_subband_spectrum_plots(
                    fig,
                    gs,
                    spec_data,
                    spec_time_axis,
                    spec_freq_axis,
                    spec_tstart,
                    spec_tend,
                    specconfig,
                    header,
                    toa=peak_time,
                    dm=cand.dm,
                    pulse_width=pulse_width,
                    snr=snr,
                    subband_matrix=subband_matrix,
                    subband_freq_axis=subband_freq_axis,
                )
            elif mode in ("standard", None, "std"):
                setup_spectrum_plots(
                    fig,
                    gs,
                    spec_data,
                    spec_time_axis,
                    spec_freq_axis,
                    spec_tstart,
                    spec_tend,
                    specconfig,
                    header,
                    toa=peak_time,
                    dm=cand.dm,
                    pulse_width=pulse_width,
                    snr=snr,
                )
            elif mode == "detrend":
                setup_detrend_spectrum_plots(
                    fig,
                    gs,
                    spec_data,
                    spec_time_axis,
                    spec_freq_axis,
                    spec_tstart,
                    spec_tend,
                    specconfig,
                    header,
                    toa=peak_time,
                    dm=cand.dm,
                    pulse_width=pulse_width,
                    snr=snr,
                )
            else:
                raise ValueError(f"Unsupported spectrum mode: {mode}")
        except Exception as exc:
            print(f"Warning: Failed to process spectrum data: {exc}")
            snr, pulse_width_ms, peak_time, ref_toa = -1, -1, cand.toa, cand.ref_toa

        basename = os.path.basename(file_path).split(".")[0]
        fig.suptitle(
            f"FILE: {basename} - DM: {cand.dm} - TOA: {ref_toa:.3f}s - SNR: {snr:.2f} - "
            f"Pulse Width: {pulse_width_ms:.2f} ms - Peak Time: {peak_time:.3f}s",
            fontsize=16,
            y=0.96,
        )

        savetype = specconfig.savetype
        if savetype == "jpg":
            imgname = f"{snr:.2f}_{pulse_width_ms:.2f}_{cand.dm}_{ref_toa:.3f}_{dmt.__str__()}.jpg"
            output_filename = f"{save_path}/{imgname}"
            print(f"Saving: {os.path.basename(output_filename)}")

            plt.savefig(output_filename, dpi=100, format="jpg", bbox_inches="tight")
        else:
            imgname = f"{snr:.2f}_{pulse_width_ms:.2f}_{cand.dm}_{ref_toa:.3f}_{dmt.__str__()}.png"
            output_filename = f"{save_path}/{imgname}"
            print(f"Saving: {os.path.basename(output_filename)}")

            plt.savefig(output_filename, dpi=dpi, format="png", bbox_inches="tight")

        if taskconfig.gencand:
            cand_info = {
                "file": os.path.basename(file_path),
                "mjd": header.mjd + (round(ref_toa, 3) / 86400.0),
                "dms": cand.dm,
                "toa": round(ref_toa, 3),
                "toa_ref_freq_end": cand.toa,
                "snr": round(snr, 2),
                "pulse_width_ms": round(pulse_width_ms, 2),
                "freq_start": cand.freq_start,
                "freq_end": cand.freq_end,
                "file_path": file_path,
                "plot_path": imgname,
            }
            candsinfopath = f"{save_path}/astroflow_cands.csv"
            save_candidate_info(candsinfopath, cand_info)

    except Exception as exc:
        print(f"Error in plot_candidate: {exc}")
        raise
    finally:
        plt.close("all")
        gc.collect()


def _resolve_maskfile(taskconfig: TaskConfig, file_path: str) -> str:
    basename = os.path.basename(file_path).split(".")[0]
    maskfile = f"{taskconfig.maskdir}/{basename}_your_rfi_mask.bad_chans"
    if not os.path.exists(maskfile):
        maskfile = taskconfig.maskfile
    return maskfile


def _close_origin_data(origin_data) -> None:
    if origin_data is None:
        return
    close_method = getattr(origin_data, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception:
            pass


def _boxcar_max_samples(specconfig, header):
    max_ms = specconfig.snr_boxcar_max_ms
    if max_ms is None:
        return None
    if max_ms <= 0:
        return None
    return max(1, int(round((max_ms * 1e-3) / header.tsamp)))


def _normalize_mode(mode):
    if mode in (None, "std"):
        return "standard"
    return mode


def _snr_input_from_subband(mode, specconfig, subband_matrix):
    if mode == "detrend":
        return detrend(subband_matrix, axis=0, trend="linear")
    if mode == "subband" and specconfig.dtrend:
        return detrend(subband_matrix, axis=0, trend="linear")
    return subband_matrix


def _resolve_subfreq(specconfig, nchan):
    subfreq = specconfig.subfreq
    if subfreq is None or subfreq <= 0:
        return nchan
    return max(1, min(int(subfreq), nchan))
