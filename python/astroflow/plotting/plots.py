from __future__ import annotations

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .analysis import calculate_frb_snr, detrend, downsample_freq_weighted_vec

AXIS_LABEL_FONTSIZE = 17
AXIS_TICK_FONTSIZE = 15
INFO_FONTSIZE = 10.3
LEGEND_FONTSIZE = 13


def _normalize_channels_for_display(data, clip_sigma=6.0, eps=1e-6):
    """Normalize each frequency channel for display to reduce stripe artifacts."""
    med = np.nanmedian(data, axis=0)
    mad = np.nanmedian(np.abs(data - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale < eps, 1.0, scale)
    norm = (data - med) / scale
    if clip_sigma is not None:
        norm = np.clip(norm, -clip_sigma, clip_sigma)
    flat = mad < eps
    if np.any(flat):
        norm[:, flat] = 0.0
    return norm


def _boxcar_max_samples(specconfig, header):
    max_ms = specconfig.snr_boxcar_max_ms
    if max_ms is None:
        return None
    if max_ms <= 0:
        return None
    return max(1, int(round((max_ms * 1e-3) / header.tsamp)))




def prepare_dm_data(dmt):
    """Prepare DM-Time data for plotting."""
    dm_data = np.array(dmt.data, dtype=np.float32)
    dm_data = cv2.cvtColor(dm_data, cv2.COLOR_BGR2GRAY)
    dm_data = cv2.normalize(dm_data, None, 0, 255, cv2.NORM_MINMAX)
    time_axis = np.linspace(dmt.tstart, dmt.tend, dm_data.shape[1])
    dm_axis = np.linspace(dmt.dm_low, dmt.dm_high, dm_data.shape[0])

    return dm_data, time_axis, dm_axis


def calculate_spectrum_time_window(toa, pulse_width_samples, tsamp, tband, multiplier=40.0):
    """Calculate spectrum time window around TOA based on pulse width."""
    if pulse_width_samples > 0:
        pulse_width_seconds = pulse_width_samples * tsamp
        time_size = multiplier * pulse_width_seconds / 2
    else:
        time_size = (tband * 1e-3) / 2

    spec_tstart = max(0, toa - time_size)
    spec_tend = toa + time_size
    return spec_tstart, spec_tend


def setup_dm_plots(fig, gs, dm_data, time_axis, dm_axis, dm_vmin, dm_vmax, dm, toa):
    """Setup DM-Time subplot components."""
    ax_time = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_time)
    ax_dm = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main DM-Time plot
    ax_main.imshow(
        dm_data,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=dm_vmin,
        vmax=dm_vmax,
        extent=[time_axis[0], time_axis[-1], dm_axis[0], dm_axis[-1]],
    )
    ax_main.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
    ax_main.set_ylabel("DM (pc cm$^{-3}$)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax_main.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE)
    ax_main.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
    ax_main.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

    # Add dashed ellipse around the candidate region.
    time_range = time_axis[-1] - time_axis[0]
    dm_range = dm_axis[-1] - dm_axis[0]
    radius_time = time_range * 0.05
    radius_dm = dm_range * 0.07

    circle = mpatches.Ellipse(
        (toa, dm),
        width=2 * radius_time,
        height=2 * radius_dm,
        fill=False,
        linestyle="--",
        linewidth=2,
        edgecolor="white",
        alpha=0.7,
        label=f"Candidate: DM={dm:.2f}, TOA={toa:.3f}s",
    )
    ax_main.add_patch(circle)

    # DM marginal plot
    dm_sum = np.max(dm_data, axis=1)
    ax_dm.plot(dm_sum, dm_axis, lw=1.5, color="darkblue")
    ax_dm.tick_params(axis="y", labelleft=False, labelsize=AXIS_TICK_FONTSIZE)
    ax_dm.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE)
    ax_dm.set_title("DM Int.", fontsize=AXIS_LABEL_FONTSIZE, pad=6)
    ax_dm.grid(alpha=0.3)

    # Time marginal plot
    time_sum = np.max(dm_data, axis=0)
    ax_time.plot(time_axis, time_sum, lw=1.5, color="darkred")
    ax_time.tick_params(axis="x", labelbottom=False, labelsize=AXIS_TICK_FONTSIZE)
    ax_time.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
    ax_time.set_ylabel("T Int.", fontsize=AXIS_LABEL_FONTSIZE, labelpad=6)
    ax_time.grid(alpha=0.3)
    ax_time.text(
        0.02,
        0.95,
        f"DM: {dm:.2f} pc $cm^{{-3}}$ \n TOA: {toa:.3f}s",
        transform=ax_time.transAxes,
        fontsize=INFO_FONTSIZE,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    return ax_time, ax_main, ax_dm


def setup_spectrum_plots(
    fig,
    gs,
    spec_data,
    spec_time_axis,
    spec_freq_axis,
    spec_tstart,
    spec_tend,
    specconfig,
    header,
    col_base=2,
    toa=None,
    dm=None,
    ref_toa=None,
    pulse_width=None,
    snr=None,
):
    """Setup standard spectrum subplot components without subband analysis."""
    ax_spec_time = fig.add_subplot(gs[0, col_base])
    ax_spec = fig.add_subplot(gs[1, col_base], sharex=ax_spec_time)
    ax_spec_freq = fig.add_subplot(gs[1, col_base + 1], sharey=ax_spec)

    time_series = np.sum(spec_data, axis=1)
    ax_spec_time.plot(spec_time_axis, time_series, "-", color="black", linewidth=1)

    if toa is not None:
        ax_spec_time.axvline(toa, color="blue", linestyle="--", linewidth=1, alpha=0.8, label=f"TOA: {toa:.3f}s")

    if snr is not None and pulse_width is not None:
        pulse_width_ms = pulse_width * header.tsamp * 1000 if pulse_width > 0 else -1
        ax_spec_time.text(
            0.02,
            0.96,
            f"SNR: {snr:.2f}\nPulse Width: {pulse_width_ms:.2f} ms",
            transform=ax_spec_time.transAxes,
            fontsize=INFO_FONTSIZE,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax_spec_time.set_ylabel("Int. Power", fontsize=AXIS_LABEL_FONTSIZE)
    ax_spec_time.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_spec_time.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
    ax_spec_time.grid(True, alpha=0.3)
    if toa is not None:
        ax_spec_time.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")

    freq_series = np.sum(spec_data, axis=0)
    vmin = freq_series[freq_series > 0].min()
    vmax = freq_series.max()
    ax_spec_freq.plot(freq_series, spec_freq_axis, "-", color="darkblue", linewidth=1)
    ax_spec_freq.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_spec_freq.grid(True, alpha=0.3)
    ax_spec_freq.set_title("Freq. Int. Power", fontsize=AXIS_LABEL_FONTSIZE, pad=6)
    ax_spec_freq.set_xlim(vmin, vmax * 1.01)
    x_min, x_max = ax_spec_freq.get_xlim()
    ax_spec_freq.set_xticks([x_min, x_max])
    ax_spec_freq.set_xticklabels(["0", "1"], fontsize=AXIS_TICK_FONTSIZE)
    ax_spec_freq.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

    ax_info = fig.add_subplot(gs[0, col_base + 1])
    ax_info.axis("off")
    info_lines = [
        f"FCH1={header.fch1:.3f} MHz",
        f"FOFF={header.foff:.3f} MHz",
        f"TSAMP={header.tsamp:.6e}s",
    ]
    if dm is not None:
        info_lines.append(f"DM={dm:.2f}")
    if ref_toa is not None:
        info_lines.append(f"ref TOA={ref_toa:.3f}s")
    ax_info.text(
        0.98,
        0.98,
        "\n".join(info_lines),
        transform=ax_info.transAxes,
        fontsize=INFO_FONTSIZE+2,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    spec_vmin = np.percentile(spec_data, specconfig.minpercentile)
    spec_vmax = np.percentile(spec_data, specconfig.maxpercentile)
    if spec_vmin == 0:
        non_zero_values = spec_data[spec_data > 1]
        if non_zero_values.size > 0:
            spec_vmin = non_zero_values.min()

    extent = [spec_time_axis[0], spec_time_axis[-1], spec_freq_axis[0], spec_freq_axis[-1]]
    ax_spec.imshow(
        spec_data.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=extent,
        vmin=spec_vmin,
        vmax=spec_vmax,
    )

    ax_spec.set_ylabel("Frequency (MHz)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_spec.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
    ax_spec.set_xlim(spec_tstart, spec_tend)
    ax_spec.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE)
    ax_spec.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
    ax_spec.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

    return ax_spec_time, ax_spec, ax_spec_freq


def setup_detrend_spectrum_plots(
    fig,
    gs,
    spec_data,
    spec_time_axis,
    spec_freq_axis,
    spec_tstart,
    spec_tend,
    specconfig,
    header,
    col_base=2,
    toa=None,
    dm=None,
    ref_toa=None,
    pulse_width=None,
    snr=None,
    detrend_type="linear",
):
    """Setup spectrum subplot components with detrending applied for better signal visibility."""
    ax_spec_time = fig.add_subplot(gs[0, col_base])
    ax_spec = fig.add_subplot(gs[1, col_base], sharex=ax_spec_time)
    ax_spec_freq = fig.add_subplot(gs[1, col_base + 1], sharey=ax_spec)

    detrend_data = spec_data.T

    try:
        detrended_freq_time = detrend(detrend_data, axis=1, trend=detrend_type)
        detrended_data = detrended_freq_time.T
    except Exception as exc:
        print(f"Detrending failed: {exc}, using original data")
        detrended_data = spec_data

    if snr is None or pulse_width is None or toa is None:
        toa_sample_idx = None if toa is None else int((toa - spec_tstart) / header.tsamp)
        snr, pulse_width, peak_idx, _ = calculate_frb_snr(
            detrended_data,
            noise_range=None,
            threshold_sigma=5,
            toa_sample_idx=toa_sample_idx,
            fitting_window_samples=_boxcar_max_samples(specconfig, header),
            tsamp=header.tsamp,
        )
        toa = spec_tstart + (peak_idx + 0.5) * header.tsamp

    display_data = _normalize_channels_for_display(detrended_data)

    time_series = np.sum(detrended_data, axis=1)
    ax_spec_time.plot(spec_time_axis, time_series, "-", color="black", linewidth=1)

    if toa is not None:
        ax_spec_time.axvline(toa, color="blue", linestyle="--", linewidth=1, alpha=0.8, label=f"TOA: {toa:.3f}s")

    info_text = f"SNR: {snr:.2f}" if snr is not None else "SNR: N/A"
    if pulse_width is not None:
        pulse_width_ms = pulse_width * header.tsamp * 1000 if pulse_width > 0 else -1
        info_text += f"\nPulse Width: {pulse_width_ms:.2f} ms"
    info_text += f"\nDetrend: {detrend_type} (per freq channel)"

    ax_spec_time.text(
        0.02,
        0.96,
        info_text,
        transform=ax_spec_time.transAxes,
        fontsize=INFO_FONTSIZE,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax_spec_time.set_ylabel("Int. Power\n(Detrend)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_spec_time.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_spec_time.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
    ax_spec_time.grid(True, alpha=0.3)
    if toa is not None:
        ax_spec_time.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")

    freq_series = np.sum(detrended_data, axis=0)
    ax_spec_freq.plot(freq_series, spec_freq_axis, "-", color="darkblue", linewidth=1)
    ax_spec_freq.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_spec_freq.grid(True, alpha=0.3)
    ax_spec_freq.set_title("Freq. Int. Power", fontsize=AXIS_LABEL_FONTSIZE, pad=6)
    x_min, x_max = ax_spec_freq.get_xlim()
    ax_spec_freq.set_xticks([x_min, x_max])
    ax_spec_freq.set_xticklabels(["0", "1"], fontsize=AXIS_TICK_FONTSIZE)
    ax_spec_freq.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

    ax_info = fig.add_subplot(gs[0, col_base + 1])
    ax_info.axis("off")
    info_lines = [
        f"Detrend: {detrend_type.title()}",
        f"FCH1={header.fch1:.3f} MHz",
        f"FOFF={header.foff:.3f} MHz",
        f"TSAMP={header.tsamp:.6e}s",
    ]
    if dm is not None:
        info_lines.append(f"DM={dm:.2f}")
    if ref_toa is not None:
        info_lines.append(f"ref TOA={ref_toa:.3f}s")
    ax_info.text(
        0.98,
        0.98,
        "\n".join(info_lines),
        transform=ax_info.transAxes,
        fontsize=INFO_FONTSIZE,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    spec_vmin = np.percentile(display_data, specconfig.minpercentile)
    spec_vmax = np.percentile(display_data, specconfig.maxpercentile)
    if spec_vmin == 0:
        non_zero_values = display_data[display_data > 0]
        if non_zero_values.size > 0:
            spec_vmin = non_zero_values.min()

    extent = [spec_time_axis[0], spec_time_axis[-1], spec_freq_axis[0], spec_freq_axis[-1]]
    ax_spec.imshow(
        display_data.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=extent,
        vmin=spec_vmin,
        vmax=spec_vmax,
    )

    ax_spec.set_ylabel("Frequency (MHz)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_spec.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
    ax_spec.set_xlim(spec_tstart, spec_tend)
    ax_spec.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE)
    ax_spec.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
    ax_spec.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

    return ax_spec_time, ax_spec, ax_spec_freq


def setup_subband_spectrum_plots(
    fig,
    gs,
    spec_data,
    spec_time_axis,
    spec_freq_axis,
    spec_tstart,
    spec_tend,
    specconfig,
    header,
    col_base=2,
    toa=None,
    dm=None,
    ref_toa=None,
    pulse_width=None,
    snr=None,
    subband_matrix=None,
    subband_freq_axis=None,
):
    """Setup spectrum subplot components with subband analysis for enhanced weak pulse visibility."""
    ax_spec_time = fig.add_subplot(gs[0, col_base])
    ax_spec = fig.add_subplot(gs[1, col_base], sharex=ax_spec_time)
    ax_spec_freq = fig.add_subplot(gs[1, col_base + 1], sharey=ax_spec)

    if subband_matrix is None:
        n_freq_subbands = specconfig.subfreq
        subband_matrix, _ = downsample_freq_weighted_vec(spec_data, spec_freq_axis, n_freq_subbands)
        subband_freq_axis = np.linspace(spec_freq_axis[0], spec_freq_axis[-1], n_freq_subbands + 1)
    elif subband_freq_axis is None:
        raise ValueError("subband_freq_axis is required when subband_matrix is provided")

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
        for f_bin in range(n_freq_subbands):
            freq_column = subband_matrix[:, f_bin]
            col_min = np.min(freq_column)
            col_max = np.max(freq_column)
            denom = col_max - col_min
            if np.isclose(denom, 0) or denom < 1e-10:
                freq_column_norm = np.zeros_like(freq_column)
            else:
                freq_column_norm = (freq_column - col_min) / denom
            subband_matrix[:, f_bin] = freq_column_norm

    subband_time_axis = np.linspace(spec_tstart, spec_tend, n_time_bins + 1)
    subband_freq_axis = np.asarray(subband_freq_axis)

    subband_time_series = np.sum(subband_matrix, axis=1)
    subband_time_centers = 0.5 * (subband_time_axis[:-1] + subband_time_axis[1:])
    ax_spec_time.plot(subband_time_centers, subband_time_series, "-", color="black", linewidth=1, alpha=0.9)

    ax_spec_time.text(
        0.02,
        0.96,
        f"SNR: {snr:.2f} \n" f"pulse width: {pulse_width * header.tsamp * 1000:.2f} ms",
        transform=ax_spec_time.transAxes,
        fontsize=INFO_FONTSIZE,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    if toa is not None:
        ax_spec_time.axvline(toa, color="blue", linestyle="--", linewidth=1, alpha=0.8, label=f"TOA: {toa:.3f}s")

    ax_spec_time.set_ylabel("Int. Power", fontsize=AXIS_LABEL_FONTSIZE)
    ax_spec_time.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_spec_time.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
    ax_spec_time.grid(True, alpha=0.3)
    ax_spec_time.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")

    subband_freq_series = np.sum(subband_matrix, axis=0)

    zero_band = np.all(np.isclose(subband_matrix, 0.0, atol=0), axis=0)
    subband_freq_series[zero_band] = np.nan

    finite_vals = np.asarray(subband_freq_series)[np.isfinite(subband_freq_series)]
    low_bound, high_bound = np.min(finite_vals), np.max(finite_vals)

    subband_freq_centers = 0.5 * (subband_freq_axis[:-1] + subband_freq_axis[1:])

    ax_spec_freq.plot(subband_freq_series, subband_freq_centers, "-", color="black", linewidth=1, alpha=0.8)
    ax_spec_freq.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_spec_freq.grid(True, alpha=0.3)
    ax_spec_freq.set_title("Freq. Int. Power", fontsize=AXIS_LABEL_FONTSIZE, pad=6)
    ax_spec_freq.set_xlim(
        low_bound - 0.1 * abs(high_bound - low_bound),
        high_bound + 0.1 * abs(high_bound - low_bound),
    )
    x_min, x_max = ax_spec_freq.get_xlim()
    ax_spec_freq.set_xticks([x_min, x_max])
    ax_spec_freq.set_xticklabels(["0", "1"], fontsize=AXIS_TICK_FONTSIZE)
    ax_spec_freq.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

    ax_info = fig.add_subplot(gs[0, col_base + 1])
    ax_info.axis("off")
    info_lines = [
        f"Subbands: {n_freq_subbands} ({freq_subband_size:.2f} chans)",
        f"Bins: {n_time_bins} ({time_bin_duration * 1000:.3f} ms)",
        f"FCH1={header.fch1:.3f} MHz",
        f"FOFF={header.foff:.3f} MHz",
        f"TSAMP={header.tsamp:.6e}s",
    ]
    if dm is not None:
        info_lines.append(f"DM={dm:.2f}")
    if ref_toa is not None:
        info_lines.append(f"ref TOA={ref_toa:.3f}s")
    ax_info.text(
        1.02,
        0.98,
        "\n".join(info_lines),
        transform=ax_info.transAxes,
        fontsize=INFO_FONTSIZE+0.3,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    extent_subband = [
        subband_time_axis[0],
        subband_time_axis[-1],
        subband_freq_axis[0],
        subband_freq_axis[-1],
    ]

    spec_vmin = np.percentile(subband_matrix, specconfig.minpercentile)
    spec_vmax = np.percentile(subband_matrix, specconfig.maxpercentile)

    ax_spec.imshow(
        subband_matrix.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=extent_subband,
        vmin=spec_vmin,
        vmax=spec_vmax,
        interpolation="nearest",
    )

    ax_spec.set_ylabel("Frequency (MHz)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_spec.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
    ax_spec.set_xlim(spec_tstart, spec_tend)
    ax_spec.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE)
    ax_spec.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
    ax_spec.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

    return ax_spec_time, ax_spec, ax_spec_freq
