from __future__ import annotations

import gc
import os

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 后端，更快

from ..config.taskconfig import TaskConfig
from ..dedispered import dedisperse_spec_with_dm
from ..utils import get_freq_end_toa
from .analysis import calculate_frb_snr, detrend, downsample_freq_weighted_vec
from .io import load_data_file, save_candidate_info
from .plots import _normalize_channels_for_display, calculate_spectrum_time_window, prepare_dm_data
from .session import (
    CandidatePlotPayload,
    CandidatePlotSession,
    DmPlotPayload,
    SpectrumPlotPayload,
)
from .types import CandidateInfo, ensure_candidate_info

_PLOTTED_FILE_COUNT = 0
_PLOT_SESSION_CACHE: dict[tuple[int, bool], CandidatePlotSession] = {}


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
        plot_candidates_for_file(
            origin_data,
            file_path,
            [(dmt, candinfo, save_path)],
            dmtconfig,
            specconfig,
            dpi,
        )
    finally:
        _close_origin_data(origin_data)
        _collect_file_gc(specconfig)


def plot_candidates_for_file(origin_data, file_path, candidates, dmtconfig, specconfig, dpi=150):
    """
    Plot multiple candidates for the same file using a shared IO handle.
    candidates: iterable of (dmt, candinfo, save_path)
    """
    if TaskConfig().onlycand:
        return save_candidate_metrics_for_file(origin_data, file_path, candidates, specconfig)

    header = origin_data.header()
    taskconfig = TaskConfig()
    maskfile = _resolve_maskfile(taskconfig, file_path)
    session = _get_plot_session(dpi, bool(getattr(specconfig, "onlyspec", False)))
    for dmt, candinfo, save_path in candidates:
        payload = _prepare_candidate_plot_payload(
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
        )
        if payload is None:
            continue
        session.render(payload)
        _save_candidate_figures(session, payload, dpi)
        _save_candidate_metadata(taskconfig, header, file_path, payload, ensure_candidate_info(candinfo))


def plot_candidates_for_path(file_path, candidates, dmtconfig, specconfig, dpi=150):
    """
    Plot multiple candidates for the same file by opening the file once in this process.
    """
    origin_data = load_data_file(file_path)
    try:
        plot_candidates_for_file(origin_data, file_path, candidates, dmtconfig, specconfig, dpi)
    finally:
        _close_origin_data(origin_data)
        _collect_file_gc(specconfig)


def save_candidate_metrics_for_path(file_path, candidates, dmtconfig, specconfig):
    origin_data = load_data_file(file_path)
    try:
        save_candidate_metrics_for_file(origin_data, file_path, candidates, specconfig)
    finally:
        _close_origin_data(origin_data)
        _collect_file_gc(specconfig)


def save_candidate_metrics_for_file(origin_data, file_path, candidates, specconfig):
    header = origin_data.header()
    taskconfig = TaskConfig()
    maskfile = _resolve_maskfile(taskconfig, file_path)
    for _dmt, candinfo, save_path in candidates:
        cand = ensure_candidate_info(candinfo)
        print(
            f"Save cand metrics: DM={cand.dm}, TOA={cand.toa}, "
            f"Freq={cand.freq_start}-{cand.freq_end} MHz, DMT Index={cand.dmt_idx}"
        )
        metrics = _calculate_candidate_metrics(
            origin_data,
            header,
            maskfile,
            cand,
            specconfig,
        )
        cand_info = {
            "file": os.path.basename(file_path),
            "mjd": header.mjd + (round(metrics["ref_toa"], 3) / 86400.0),
            "dms": cand.dm,
            "toa": round(metrics["ref_toa"], 3),
            "toa_ref_freq_end": cand.toa,
            "snr": round(metrics["snr"], 2),
            "pulse_width_ms": round(metrics["pulse_width_ms"], 2),
            "freq_start": cand.freq_start,
            "freq_end": cand.freq_end,
            "file_path": file_path,
            "plot_path": "",
            "peak_toa": round(metrics["peak_toa"], 3),
        }
        candsinfopath = os.path.join(save_path, "astroflow_cands.csv")
        os.makedirs(save_path, exist_ok=True)
        save_candidate_info(candsinfopath, cand_info)


def _prepare_candidate_plot_payload(
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
) -> CandidatePlotPayload | None:
    cand = ensure_candidate_info(candinfo)
    print(
        f"Plot cand: DM={cand.dm}, TOA={cand.toa}, "
        f"Freq={cand.freq_start}-{cand.freq_end} MHz, DMT Index={cand.dmt_idx}"
    )

    dm_payload = _prepare_dm_payload(dmt, dmtconfig, cand)
    mode = _normalize_mode(specconfig.mode)
    ref_toa = cand.ref_toa
    peak_time = cand.toa
    snr = -1
    pulse_width = -1
    pulse_width_ms = -1
    spectrum_payload = None

    try:
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

        initial_spec_freq_axis = np.linspace(cand.freq_start, cand.freq_end, initial_spectrum.nchans)
        initial_subfreq = _resolve_subfreq(specconfig, initial_spec_data.shape[1])
        initial_subband_matrix, _ = downsample_freq_weighted_vec(
            initial_spec_data, initial_spec_freq_axis, initial_subfreq
        )
        initial_snr_input = _snr_input_from_subband(mode, specconfig, initial_subband_matrix)
        snr, pulse_width, peak_idx, _ = calculate_frb_snr(
            initial_snr_input,
            noise_range=None,
            threshold_sigma=5,
            toa_sample_idx=toa_sample_idx,
            fitting_window_samples=_boxcar_max_samples(specconfig, header),
            tsamp=header.tsamp,
        )
        if snr < taskconfig.snrhold:
            return None

        peak_time = initial_spec_tstart + (peak_idx + 0.5) * header.tsamp
        spec_tstart, spec_tend = calculate_spectrum_time_window(
            peak_time, pulse_width, header.tsamp, tband, 70
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
        pulse_width_ms = pulse_width * header.tsamp * 1e3 if pulse_width > 0 else -1
        spectrum_payload = _prepare_spectrum_payload(
            mode,
            spec_data,
            spec_time_axis,
            spec_freq_axis,
            spec_tstart,
            spec_tend,
            specconfig,
            header,
            cand.dm,
            ref_toa,
            pulse_width,
            pulse_width_ms,
            snr,
            peak_time,
        )
    except Exception as exc:
        print(f"Warning: Failed to process spectrum data: {exc}")

    basename = os.path.basename(file_path).split(".")[0]
    title, title_y = _candidate_title(basename, cand, ref_toa, snr, pulse_width_ms, peak_time)
    savetype = specconfig.savetype
    suffix = "jpg" if savetype == "jpg" else "png"
    base_name = f"{snr:.2f}_{pulse_width_ms:.2f}_{cand.dm}_{ref_toa:.3f}_{dmt.__str__()}"
    if bool(getattr(specconfig, "onlyspec", False)):
        imgname = f"{base_name}_spec.{suffix}"
        dm_imgname = f"{base_name}_dmtime.{suffix}"
    else:
        imgname = f"{base_name}.{suffix}"
        dm_imgname = None

    return CandidatePlotPayload(
        dm_plot=dm_payload,
        spectrum_plot=spectrum_payload,
        title=title,
        title_y=title_y,
        imgname=imgname,
        dm_imgname=dm_imgname,
        save_path=save_path,
        savetype=savetype,
        snr=snr,
        pulse_width_ms=pulse_width_ms,
        ref_toa=ref_toa,
        peak_toa=peak_time,
    )


def _calculate_candidate_metrics(origin_data, header, maskfile, cand: CandidateInfo, specconfig) -> dict[str, float]:
    ref_toa = cand.ref_toa
    peak_time = cand.toa
    snr = -1.0
    pulse_width_ms = -1.0
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

        initial_spec_freq_axis = np.linspace(cand.freq_start, cand.freq_end, initial_spectrum.nchans)
        initial_subfreq = _resolve_subfreq(specconfig, initial_spec_data.shape[1])
        initial_subband_matrix, _ = downsample_freq_weighted_vec(
            initial_spec_data, initial_spec_freq_axis, initial_subfreq
        )
        initial_snr_input = _snr_input_from_subband(mode, specconfig, initial_subband_matrix)
        snr, pulse_width, peak_idx, _ = calculate_frb_snr(
            initial_snr_input,
            noise_range=None,
            threshold_sigma=5,
            toa_sample_idx=toa_sample_idx,
            fitting_window_samples=_boxcar_max_samples(specconfig, header),
            tsamp=header.tsamp,
        )
        peak_time = initial_spec_tstart + (peak_idx + 0.5) * header.tsamp
        pulse_width_ms = pulse_width * header.tsamp * 1e3 if pulse_width > 0 else -1
    except Exception as exc:
        print(f"Warning: Failed to calculate candidate metrics: {exc}")

    return {
        "ref_toa": float(ref_toa),
        "snr": float(snr),
        "pulse_width_ms": float(pulse_width_ms),
        "peak_toa": float(peak_time),
    }


def _prepare_dm_payload(dmt, dmtconfig, cand: CandidateInfo) -> DmPlotPayload:
    dm_data, time_axis, dm_axis = prepare_dm_data(dmt)
    dm_vmin, dm_vmax = np.percentile(
        dm_data, [dmtconfig.minpercentile, dmtconfig.maxpercentile]
    )
    return DmPlotPayload(
        data=dm_data,
        time_axis=time_axis,
        dm_axis=dm_axis,
        vmin=dm_vmin,
        vmax=dm_vmax,
        dm=cand.dm,
        toa=cand.toa,
    )


def _prepare_spectrum_payload(
    mode,
    spec_data,
    spec_time_axis,
    spec_freq_axis,
    spec_tstart,
    spec_tend,
    specconfig,
    header,
    dm,
    ref_toa,
    pulse_width,
    pulse_width_ms,
    snr,
    peak_time,
) -> SpectrumPlotPayload:
    if mode == "subband":
        return _prepare_subband_payload(
            spec_data,
            spec_freq_axis,
            spec_tstart,
            spec_tend,
            specconfig,
            header,
            dm,
            ref_toa,
            pulse_width,
            pulse_width_ms,
            snr,
            peak_time,
        )
    if mode == "standard":
        return _prepare_standard_payload(
            spec_data,
            spec_time_axis,
            spec_freq_axis,
            spec_tstart,
            spec_tend,
            specconfig,
            header,
            dm,
            ref_toa,
            pulse_width_ms,
            snr,
            peak_time,
        )
    if mode == "detrend":
        return _prepare_detrend_payload(
            spec_data,
            spec_time_axis,
            spec_freq_axis,
            spec_tstart,
            spec_tend,
            specconfig,
            header,
            dm,
            ref_toa,
            pulse_width_ms,
            snr,
            peak_time,
        )
    raise ValueError(f"Unsupported spectrum mode: {mode}")


def _prepare_standard_payload(
    spec_data,
    spec_time_axis,
    spec_freq_axis,
    spec_tstart,
    spec_tend,
    specconfig,
    header,
    dm,
    ref_toa,
    pulse_width_ms,
    snr,
    peak_time,
) -> SpectrumPlotPayload:
    spec_vmin, spec_vmax = _image_percentiles(spec_data, specconfig)
    if spec_vmin == 0:
        non_zero_values = spec_data[spec_data > 1]
        if non_zero_values.size > 0:
            spec_vmin = non_zero_values.min()

    freq_series = np.sum(spec_data, axis=0)
    freq_xlim = _series_bounds(freq_series, positive_min=True)
    return SpectrumPlotPayload(
        image=spec_data,
        extent=[spec_time_axis[0], spec_time_axis[-1], spec_freq_axis[0], spec_freq_axis[-1]],
        vmin=spec_vmin,
        vmax=spec_vmax,
        xlim=(spec_tstart, spec_tend),
        ylim=(spec_freq_axis[0], spec_freq_axis[-1]),
        time_x=spec_time_axis,
        time_y=np.sum(spec_data, axis=1),
        freq_x=freq_series,
        freq_y=spec_freq_axis,
        freq_xlim=freq_xlim,
        time_info=f"SNR: {snr:.2f}\nPulse Width: {pulse_width_ms:.2f} ms",
        info_text=_spectrum_info_text(header, dm, ref_toa),
        toa=peak_time,
        freq_color="darkblue",
        interpolation="auto",
    )


def _prepare_detrend_payload(
    spec_data,
    spec_time_axis,
    spec_freq_axis,
    spec_tstart,
    spec_tend,
    specconfig,
    header,
    dm,
    ref_toa,
    pulse_width_ms,
    snr,
    peak_time,
) -> SpectrumPlotPayload:
    try:
        detrended_data = detrend(spec_data.T, axis=1, trend="linear").T
    except Exception as exc:
        print(f"Detrending failed: {exc}, using original data")
        detrended_data = spec_data

    display_data = _normalize_channels_for_display(detrended_data)
    spec_vmin, spec_vmax = _image_percentiles(display_data, specconfig)
    if spec_vmin == 0:
        non_zero_values = display_data[display_data > 0]
        if non_zero_values.size > 0:
            spec_vmin = non_zero_values.min()

    freq_series = np.sum(detrended_data, axis=0)
    return SpectrumPlotPayload(
        image=display_data,
        extent=[spec_time_axis[0], spec_time_axis[-1], spec_freq_axis[0], spec_freq_axis[-1]],
        vmin=spec_vmin,
        vmax=spec_vmax,
        xlim=(spec_tstart, spec_tend),
        ylim=(spec_freq_axis[0], spec_freq_axis[-1]),
        time_x=spec_time_axis,
        time_y=np.sum(detrended_data, axis=1),
        freq_x=freq_series,
        freq_y=spec_freq_axis,
        freq_xlim=_series_bounds(freq_series),
        time_info=f"SNR: {snr:.2f}\nPulse Width: {pulse_width_ms:.2f} ms\nDetrend: linear (per freq channel)",
        info_text=_spectrum_info_text(header, dm, ref_toa, prefix="Detrend: Linear"),
        toa=peak_time,
        time_ylabel="Int. Power\n(Detrend)",
        freq_color="darkblue",
        interpolation="auto",
    )


def _prepare_subband_payload(
    spec_data,
    spec_freq_axis,
    spec_tstart,
    spec_tend,
    specconfig,
    header,
    dm,
    ref_toa,
    pulse_width,
    pulse_width_ms,
    snr,
    peak_time,
) -> SpectrumPlotPayload:
    subfreq = _resolve_subfreq(specconfig, spec_data.shape[1])
    subband_matrix, _ = downsample_freq_weighted_vec(spec_data, spec_freq_axis, subfreq)
    subband_freq_axis = np.linspace(spec_freq_axis[0], spec_freq_axis[-1], subfreq + 1)

    n_time_samples, n_freq_subbands = subband_matrix.shape
    subtsamp = max(1, int(specconfig.subtsamp))
    if pulse_width and pulse_width > 0:
        time_bin_size = max(1, int(round(pulse_width / subtsamp)))
    else:
        time_bin_size = subtsamp
    if time_bin_size > n_time_samples:
        time_bin_size = n_time_samples
    n_time_bins = max(1, n_time_samples // time_bin_size)
    trimmed_time_len = n_time_bins * time_bin_size
    time_bin_duration = time_bin_size * header.tsamp
    freq_subband_size = max(1, len(spec_freq_axis) / n_freq_subbands)

    if trimmed_time_len < n_time_samples:
        subband_matrix = subband_matrix[:trimmed_time_len, :]
    if time_bin_size > 1:
        subband_matrix = subband_matrix.reshape(
            n_time_bins, time_bin_size, n_freq_subbands
        ).sum(axis=1)

    if specconfig.dtrend:
        subband_matrix = detrend(subband_matrix, axis=0, trend="linear")

    if specconfig.norm:
        col_min = np.min(subband_matrix, axis=0)
        col_max = np.max(subband_matrix, axis=0)
        denom = col_max - col_min
        valid = (~np.isclose(denom, 0)) & (denom >= 1e-10)
        normalized = np.zeros_like(subband_matrix)
        normalized[:, valid] = (subband_matrix[:, valid] - col_min[valid]) / denom[valid]
        subband_matrix = normalized

    subband_time_axis = np.linspace(spec_tstart, spec_tend, n_time_bins + 1)
    subband_freq_axis = np.asarray(subband_freq_axis)
    subband_time_centers = 0.5 * (subband_time_axis[:-1] + subband_time_axis[1:])
    subband_freq_centers = 0.5 * (subband_freq_axis[:-1] + subband_freq_axis[1:])
    subband_freq_series = np.sum(subband_matrix, axis=0)
    zero_band = np.all(np.isclose(subband_matrix, 0.0, atol=0), axis=0)
    subband_freq_series[zero_band] = np.nan

    spec_vmin, spec_vmax = _image_percentiles(subband_matrix, specconfig)
    info_lines = [
        f"Subbands: {n_freq_subbands} ({freq_subband_size:.2f} chans)",
        f"Bins: {n_time_bins} ({time_bin_duration * 1000:.3f} ms)",
        f"FCH1={header.fch1:.3f} MHz",
        f"FOFF={header.foff:.3f} MHz",
        f"TSAMP={header.tsamp:.6e}s",
        f"DM={dm:.2f}",
        f"ref TOA={ref_toa:.3f}s",
    ]
    return SpectrumPlotPayload(
        image=subband_matrix,
        extent=[subband_time_axis[0], subband_time_axis[-1], subband_freq_axis[0], subband_freq_axis[-1]],
        vmin=spec_vmin,
        vmax=spec_vmax,
        xlim=(spec_tstart, spec_tend),
        ylim=(subband_freq_axis[0], subband_freq_axis[-1]),
        time_x=subband_time_centers,
        time_y=np.sum(subband_matrix, axis=1),
        freq_x=subband_freq_series,
        freq_y=subband_freq_centers,
        freq_xlim=_series_bounds(subband_freq_series),
        time_info=f"SNR: {snr:.2f} \n" f"pulse width: {pulse_width_ms:.2f} ms",
        info_text="\n".join(info_lines),
        toa=peak_time,
    )


def _candidate_title(basename, cand: CandidateInfo, ref_toa, snr, pulse_width_ms, peak_time):
    metrics_title = (
        f"DM: {cand.dm} - TOA: {ref_toa:.3f}s - SNR: {snr:.2f} - "
        f"Pulse Width: {pulse_width_ms:.2f} ms - Peak Time: {peak_time:.3f}s"
    )
    if len(basename) > 55:
        return f"FILE: {basename}\n{metrics_title}", 0.985
    return f"FILE: {basename} - {metrics_title}", 0.94


def _spectrum_info_text(header, dm, ref_toa, prefix=None) -> str:
    info_lines = []
    if prefix:
        info_lines.append(prefix)
    info_lines.extend(
        [
            f"FCH1={header.fch1:.3f} MHz",
            f"FOFF={header.foff:.3f} MHz",
            f"TSAMP={header.tsamp:.6e}s",
            f"DM={dm:.2f}",
            f"ref TOA={ref_toa:.3f}s",
        ]
    )
    return "\n".join(info_lines)


def _image_percentiles(data, specconfig):
    return np.percentile(data, [specconfig.minpercentile, specconfig.maxpercentile])


def _series_bounds(values, positive_min=False) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if positive_min:
        positive = arr[arr > 0]
        if positive.size > 0:
            low = float(np.min(positive))
            high = float(np.max(arr)) if arr.size else low + 1
            return _padded_bounds(low, high, lower_pad=0.0, upper_frac=0.01)
    if arr.size == 0:
        return (0.0, 1.0)
    return _padded_bounds(float(np.min(arr)), float(np.max(arr)))


def _padded_bounds(low, high, lower_pad=None, upper_frac=0.1) -> tuple[float, float]:
    if not np.isfinite(low) or not np.isfinite(high):
        return (0.0, 1.0)
    if np.isclose(low, high):
        pad = max(abs(high) * 0.05, 1.0)
        return (low - pad, high + pad)
    if lower_pad is not None:
        return (low - lower_pad, high + upper_frac * abs(high))
    pad = 0.1 * abs(high - low)
    return (low - pad, high + pad)


def _save_candidate_figures(session: CandidatePlotSession, payload: CandidatePlotPayload, dpi: int) -> None:
    output_filename = os.path.join(payload.save_path, payload.imgname)
    print(f"Saving: {os.path.basename(output_filename)}")
    if payload.savetype == "jpg":
        session.fig.savefig(
            output_filename,
            format="jpg",
            pil_kwargs={"quality": 85},
            dpi=dpi,
            bbox_inches=None,
            # pad_inches=0.03,
            facecolor="white",
            edgecolor="none",
        )
        if session.onlyspec and payload.dm_imgname is not None:
            dm_output_filename = os.path.join(payload.save_path, payload.dm_imgname)
            print(f"Saving: {os.path.basename(dm_output_filename)}")
            session.dm_fig.savefig(
                dm_output_filename,
                format="jpg",
                pil_kwargs={"quality": 85},
                dpi=dpi,
                bbox_inches=None,
                # pad_inches=0.03,
                facecolor="white",
                edgecolor="none",
            )
        return

    session.fig.savefig(
        output_filename,
        format="png",
        dpi=dpi,
        bbox_inches=None,
        # pad_inches=0.03,
        facecolor="white",
        edgecolor="none",
        pil_kwargs={"compress_level": 3},
    )
    if session.onlyspec and payload.dm_imgname is not None:
        dm_output_filename = os.path.join(payload.save_path, payload.dm_imgname)
        print(f"Saving: {os.path.basename(dm_output_filename)}")
        session.dm_fig.savefig(
            dm_output_filename,
            format="png",
            dpi=dpi,
            bbox_inches=None,
            # pad_inches=0.03,
            facecolor="white",
            edgecolor="none",
            pil_kwargs={"compress_level": 3},
        )


def _save_candidate_metadata(taskconfig, header, file_path, payload: CandidatePlotPayload, cand: CandidateInfo) -> None:
    if not taskconfig.gencand:
        return
    cand_info = {
        "file": os.path.basename(file_path),
        "mjd": header.mjd + (round(payload.ref_toa, 3) / 86400.0),
        "dms": cand.dm,
        "toa": round(payload.ref_toa, 3),
        "toa_ref_freq_end": cand.toa,
        "snr": round(payload.snr, 2),
        "pulse_width_ms": round(payload.pulse_width_ms, 2),
        "freq_start": cand.freq_start,
        "freq_end": cand.freq_end,
        "file_path": file_path,
        "plot_path": payload.imgname,
        "peak_toa": round(payload.peak_toa, 3),
    }
    candsinfopath = os.path.join(payload.save_path, "astroflow_cands.csv")
    save_candidate_info(candsinfopath, cand_info)


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


def _get_plot_session(dpi: int, onlyspec: bool) -> CandidatePlotSession:
    key = (int(dpi), bool(onlyspec))
    session = _PLOT_SESSION_CACHE.get(key)
    if session is None:
        session = CandidatePlotSession(dpi=key[0], onlyspec=key[1])
        _PLOT_SESSION_CACHE[key] = session
    return session


def close_plot_sessions() -> None:
    for session in _PLOT_SESSION_CACHE.values():
        session.close()
    _PLOT_SESSION_CACHE.clear()


def _collect_file_gc(specconfig) -> None:
    global _PLOTTED_FILE_COUNT
    _PLOTTED_FILE_COUNT += 1
    every = int(getattr(specconfig, "gc_collect_every_files", 10) or 0)
    if every > 0 and _PLOTTED_FILE_COUNT % every == 0:
        gc.collect()


def _boxcar_max_samples(specconfig, header):
    max_ms = specconfig.snr_boxcar_max_ms
    if max_ms is None:
        return 30
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
