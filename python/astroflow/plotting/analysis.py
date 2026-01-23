from __future__ import annotations

import numpy as np
from numpy.polynomial import Chebyshev


def _robust_mean_std(data, sigma=5.0, max_iter=3):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 1.0

    clipped = data
    for _ in range(max_iter):
        med = np.median(clipped)
        mad = np.median(np.abs(clipped - med))
        std = 1.4826 * mad if mad > 0 else np.std(clipped)
        if std <= 0:
            break
        next_clipped = clipped[np.abs(clipped - med) <= sigma * std]
        if next_clipped.size == clipped.size:
            break
        if next_clipped.size < max(10, int(0.2 * clipped.size)):
            break
        clipped = next_clipped

    mean = np.mean(clipped) if clipped.size else np.mean(data)
    std = np.std(clipped) if clipped.size else np.std(data)
    if std <= 0:
        std = np.std(data)
    if std <= 0:
        std = 1.0
    return mean, std


def _build_widths(max_width):
    max_width = int(max_width)
    if max_width <= 1:
        return [1]
    if max_width <= 32:
        return list(range(1, max_width + 1))

    small = list(range(1, 33))
    logspace = np.unique(
        np.round(np.logspace(np.log10(33), np.log10(max_width), num=12)).astype(int)
    )
    widths = small + [int(w) for w in logspace if 33 <= w <= max_width]
    return widths


def calculate_frb_snr(
    spec,
    noise_range=None,
    threshold_sigma=5.0,
    toa_sample_idx=None,
    fitting_window_samples=None,
):
    """
    Robust FRB SNR calculation with multi-scale boxcar search and off-pulse noise estimation.

    Args:
        spec: 2D spectrum array (time x frequency)
        noise_range: List of (start, end) tuples for noise regions, or None for auto-detection
        threshold_sigma: Threshold for outlier detection in baseline estimation
        toa_sample_idx: Expected TOA sample index for centered fitting
        fitting_window_samples: Max width for boxcar search (default: auto)

    Returns:
        tuple: (snr, pulse_width_samples, peak_idx_fit, (noise_mean, noise_std, fit_quality))
    """
    time_series_raw = np.sum(spec, axis=1)
    n_time = len(time_series_raw)
    if n_time == 0:
        return -1, 0, 0, (0.0, 1.0, {"fit_converged": False})

    if fitting_window_samples is None:
        max_width = max(8, min(n_time // 4, 512))
    else:
        max_width = max(4, min(int(fitting_window_samples), n_time // 2))
    max_width = max(1, min(max_width, n_time))

    if toa_sample_idx is not None:
        search_radius = max(max_width, 100)
        search_start = max(0, toa_sample_idx - search_radius)
        search_end = min(n_time, toa_sample_idx + search_radius)
    else:
        search_start = 0
        search_end = n_time

    if noise_range:
        noise_data = np.concatenate([time_series_raw[slice(start, end)] for (start, end) in noise_range])
    else:
        noise_data = None
        if toa_sample_idx is not None and (search_start > 0 or search_end < n_time):
            noise_chunks = []
            if search_start > 0:
                noise_chunks.append(time_series_raw[:search_start])
            if search_end < n_time:
                noise_chunks.append(time_series_raw[search_end:])
            if noise_chunks:
                noise_data = np.concatenate(noise_chunks)
        if noise_data is None or noise_data.size < 10:
            noise_data = time_series_raw

    noise_mean, noise_std = _robust_mean_std(noise_data, sigma=threshold_sigma)

    widths = _build_widths(max_width)
    cumsum = np.cumsum(np.insert(time_series_raw, 0, 0.0))

    best_snr = -np.inf
    best_width = 1
    best_center = int(np.argmax(time_series_raw))
    best_sum = None

    for width in widths:
        if width >= n_time:
            break
        window_sums = cumsum[width:] - cumsum[:-width]
        centers = np.arange(width // 2, width // 2 + window_sums.size)

        if toa_sample_idx is not None:
            mask = (centers >= search_start) & (centers < search_end)
            if not np.any(mask):
                continue
            sums = window_sums[mask]
            centers_sel = centers[mask]
        else:
            sums = window_sums
            centers_sel = centers

        if noise_std <= 0:
            snr_series = np.full_like(sums, -np.inf, dtype=np.float64)
        else:
            snr_series = (sums - noise_mean * width) / (noise_std * np.sqrt(width))

        local_idx = int(np.argmax(snr_series))
        local_snr = float(snr_series[local_idx])
        if local_snr > best_snr:
            best_snr = local_snr
            best_width = int(width)
            best_center = int(centers_sel[local_idx])
            best_sum = float(sums[local_idx])

    if best_sum is None:
        noise_mean, noise_std = _robust_mean_std(time_series_raw, sigma=threshold_sigma)
        peak_idx = int(np.argmax(time_series_raw))
        snr = (time_series_raw[peak_idx] - noise_mean) / noise_std if noise_std > 0 else -1
        fit_quality = {"fit_converged": False, "method": "fallback"}
        return snr, 1, peak_idx, (noise_mean, noise_std, fit_quality)

    refine_left = max(0, best_center - 2 * best_width)
    refine_right = min(n_time, best_center + 2 * best_width + 1)
    noise_mask = np.ones(n_time, dtype=bool)
    noise_mask[refine_left:refine_right] = False
    refine_data = time_series_raw[noise_mask]
    if refine_data.size >= max(10, int(0.1 * n_time)):
        noise_mean, noise_std = _robust_mean_std(refine_data, sigma=threshold_sigma)

    snr = (best_sum - noise_mean * best_width) / (noise_std * np.sqrt(best_width)) if noise_std > 0 else -1

    fit_quality = {
        "fit_converged": True,
        "method": "boxcar",
        "width_samples": best_width,
        "search_start": int(search_start),
        "search_end": int(search_end),
    }

    return snr, best_width, best_center, (noise_mean, noise_std, fit_quality)


def detrend_frequency(data: np.ndarray, poly_order: int = 6) -> np.ndarray:
    """
    Remove bandpass shape along frequency axis for each time sample using Chebyshev fit.

    Parameters
    ----------
    data : np.ndarray
        Spectrum data in (time, freq) format
    poly_order : int
        Order of Chebyshev polynomial for fitting

    Returns
    -------
    np.ndarray
        Flattened data with frequency baseline removed
    """
    ntime, nchan = data.shape
    # Normalize frequency channel to [-1, 1] for numerical stability.
    x = np.linspace(-1, 1, nchan)
    detrended = np.zeros_like(data)

    for t in range(ntime):
        y = data[t, :]
        valid = np.isfinite(y) & (y != 0)
        if valid.sum() > poly_order + 2:
            cheb = Chebyshev.fit(x[valid], y[valid], deg=poly_order, domain=[-1, 1])
            fitted = cheb(x)
            detrended[t, :] = y - fitted
        else:
            detrended[t, :] = y  # Skip fit if too few valid points.

    return detrended


def detrend(data: np.ndarray, axis: int = -1, trend: str = "linear", bp=0, overwrite_data: bool = False) -> np.ndarray:
    """Remove linear or constant trend along axis from data.

    Simplified version of scipy.signal.detrend for spectrum data preprocessing.

    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    trend : {'linear', 'constant'}, optional
        The type of detrending. If ``trend == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``trend == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`. This parameter
        only has an effect when ``trend == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.
    """
    from scipy import linalg

    if trend not in ["linear", "l", "constant", "c"]:
        raise ValueError("Trend type must be 'linear' or 'constant'.")

    data = np.asarray(data)
    dtype = data.dtype.char
    if dtype not in "dfDF":
        dtype = "d"

    if trend in ["constant", "c"]:
        ret = data - np.mean(data, axis, keepdims=True)
        return ret

    dshape = data.shape
    size = dshape[axis]
    bp = np.asarray(bp)
    bp = np.sort(np.unique(np.concatenate(np.atleast_1d(0, bp, size))))
    if np.any(bp > size):
        raise ValueError("Breakpoints must be less than length of data along given axis.")

    # Restructure data so that axis is along first dimension and
    # all other dimensions are collapsed into second dimension.
    rnk = len(dshape)
    if axis < 0:
        axis = axis + rnk
    newdata = np.moveaxis(data, axis, 0)
    newdata_shape = newdata.shape
    newdata = newdata.reshape(size, -1)

    if not overwrite_data:
        newdata = newdata.copy()
    if newdata.dtype.char not in "dfDF":
        newdata = newdata.astype(dtype)

    # Find leastsq fit and remove it for each piece.
    for idx in range(len(bp) - 1):
        npts = bp[idx + 1] - bp[idx]
        a_mat = np.ones((npts, 2), dtype)
        a_mat[:, 0] = np.arange(1, npts + 1, dtype=dtype) / npts
        sl = slice(bp[idx], bp[idx + 1])
        coef, _, _, _ = linalg.lstsq(a_mat, newdata[sl])
        newdata[sl] = newdata[sl] - a_mat @ coef

    newdata = newdata.reshape(newdata_shape)
    ret = np.moveaxis(newdata, 0, axis)
    return ret


def downsample_freq_weighted_vec(spec_data, freq_axis, n_out):
    """
    Fully vectorized frequency-direction downsampling.
    Preserves energy and avoids frequency drift.

    Parameters
    ----------
    spec_data : ndarray
        [ntime, nfreq_in] dynamic spectrum
    freq_axis : ndarray
        Frequency centers (ascending).
    n_out : int
        Target subband count.
    """
    ntime, nfreq_in = spec_data.shape

    # Original channel edges.
    f_edges_in = np.concatenate(
        (
            [freq_axis[0] - (freq_axis[1] - freq_axis[0]) / 2],
            0.5 * (freq_axis[:-1] + freq_axis[1:]),
            [freq_axis[-1] + (freq_axis[-1] - freq_axis[-2]) / 2],
        )
    )
    widths_in = np.diff(f_edges_in)

    # Target channel edges.
    f_edges_out = np.linspace(f_edges_in[0], f_edges_in[-1], n_out + 1)
    freq_out = 0.5 * (f_edges_out[:-1] + f_edges_out[1:])

    # Overlap matrix (n_out x nfreq_in).
    lo = np.maximum.outer(f_edges_out[:-1], f_edges_in[:-1])
    hi = np.minimum.outer(f_edges_out[1:], f_edges_in[1:])
    overlap = np.clip(hi - lo, 0, None)

    # Normalize weight matrix.
    weights = overlap / widths_in[np.newaxis, :]
    weights /= np.sum(weights, axis=1, keepdims=True)

    # Vectorized matrix multiply (preserve energy).
    spec_out = spec_data @ weights.T
    return spec_out.astype(np.float32), freq_out
