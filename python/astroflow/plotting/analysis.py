from __future__ import annotations

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.optimize import curve_fit


def gaussian(x, amp, mu, sigma, baseline):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + baseline


def calculate_frb_snr(
    spec,
    noise_range=None,
    threshold_sigma=5.0,
    toa_sample_idx=None,
    fitting_window_samples=None,
):
    """
    Professional FRB SNR calculation with TOA-centered fitting and weighted baseline estimation.

    Args:
        spec: 2D spectrum array (time x frequency)
        noise_range: List of (start, end) tuples for noise regions, or None for auto-detection
        threshold_sigma: Threshold for outlier detection in baseline estimation
        toa_sample_idx: Expected TOA sample index for centered fitting
        fitting_window_samples: Number of samples around TOA for fitting (default: auto)

    Returns:
        tuple: (snr, pulse_width_samples, peak_idx_fit, (noise_mean, noise_std, fit_quality))
    """
    # Step 1: Frequency-integrated time series
    # Use raw time series for statistics and final SNR calculation to ensure correctness
    time_series_raw = np.sum(spec, axis=1)  # Sum over frequency axis

    n_time = len(time_series_raw)
    x = np.arange(n_time)

    # Step 1.5: Refine Peak Position (Global or Local Search)
    # First, estimate a rough baseline to ensure we are finding the signal peak, not just high baseline
    global_median = np.median(time_series_raw)
    time_series_detrended = time_series_raw - global_median

    if fitting_window_samples is None:
        fitting_window_samples = max(50, int(0.2 * n_time))

    # Determine search range for the peak
    if toa_sample_idx is not None:
        # Search within a wider window around the provided TOA to correct for offsets
        # Use 3x the fitting window size for search, or at least +/- 100 samples
        search_radius = max(fitting_window_samples, 100)
        search_start = max(0, toa_sample_idx - search_radius)
        search_end = min(n_time, toa_sample_idx + search_radius)
    else:
        search_start = 0
        search_end = n_time

    # Find peak in the search region on DETRENDED data
    search_region = slice(search_start, search_end)
    if search_end > search_start:
        local_peak_idx = np.argmax(time_series_detrended[search_region])
        refined_peak_idx = search_start + local_peak_idx
    else:
        refined_peak_idx = toa_sample_idx if toa_sample_idx is not None else n_time // 2

    # Step 2: Determine fitting region centered on REFINED peak
    fit_start = max(0, refined_peak_idx - fitting_window_samples // 2)
    fit_end = min(n_time, refined_peak_idx + fitting_window_samples // 2)

    fitting_region = slice(fit_start, fit_end)
    x_fit = x[fitting_region]
    y_fit = time_series_raw[fitting_region]  # Use RAW for fitting

    # Step 3: Robust baseline estimation using RAW data
    if noise_range is None:
        # Define noise regions excluding the central fitting area
        noise_margin = max(10, int(0.1 * n_time))
        central_start = max(0, fit_start - noise_margin)
        central_end = min(n_time, fit_end + noise_margin)

        noise_regions = []
        if central_start > 0:
            noise_regions.append(slice(0, central_start))
        if central_end < n_time:
            noise_regions.append(slice(central_end, n_time))
    else:
        noise_regions = [slice(start, end) for (start, end) in noise_range]

    if noise_regions:
        noise_data = np.concatenate([time_series_raw[region] for region in noise_regions])
    else:
        # Fallback: use edge regions
        edge_size = max(5, n_time // 10)
        noise_data = np.concatenate([time_series_raw[:edge_size], time_series_raw[-edge_size:]])

    # Robust baseline estimation using median and MAD on RAW data
    noise_median = np.median(noise_data)
    noise_mad = np.median(np.abs(noise_data - noise_median))
    noise_std_robust = 1.4826 * noise_mad  # Convert MAD to std estimate

    # Remove outliers for cleaner baseline
    outlier_mask = np.abs(noise_data - noise_median) < threshold_sigma * noise_std_robust
    if np.sum(outlier_mask) > len(noise_data) * 0.5:  # Keep at least 50% of data
        clean_noise = noise_data[outlier_mask]
        noise_mean = np.mean(clean_noise)
        noise_std = np.std(clean_noise)
    else:
        noise_mean = noise_median
        noise_std = noise_std_robust

    # Step 4: Weighted Gaussian fitting on RAW data
    # Subtract baseline from fitting data
    y_fit_corrected = y_fit - noise_mean

    # Initial parameter estimation
    peak_idx_local = np.argmax(y_fit_corrected)
    peak_idx_global = fit_start + peak_idx_local

    amp0 = y_fit_corrected[peak_idx_local]
    mu0 = x_fit[peak_idx_local]  # Peak position in global coordinates

    # Estimate sigma from FWHM using moment analysis
    try:
        # Calculate second moment for width estimation
        weights = np.maximum(0, y_fit_corrected)
        if np.sum(weights) > 0:
            weighted_mean = np.average(x_fit, weights=weights)
            weighted_var = np.average((x_fit - weighted_mean) ** 2, weights=weights)
            sigma0 = np.sqrt(weighted_var)
        else:
            sigma0 = fitting_window_samples / 6  # Fallback
    except Exception:
        sigma0 = fitting_window_samples / 6

    # Ensure reasonable bounds for sigma
    sigma0 = max(0.5, min(sigma0, fitting_window_samples / 3))  # Allow smaller sigma for narrow pulses
    baseline0 = noise_mean

    # Setup fitting parameters and bounds
    p0 = [amp0, mu0, sigma0, baseline0]

    # Conservative bounds to prevent overfitting
    sigma_min = 0.1  # Allow very narrow pulses
    sigma_max = min(fitting_window_samples / 2, n_time / 4)
    mu_min = fit_start
    mu_max = fit_end - 1

    bounds = (
        [0, mu_min, sigma_min, noise_mean - 3 * noise_std],  # lower bounds
        [amp0 * 3, mu_max, sigma_max, noise_mean + 3 * noise_std],  # upper bounds
    )

    # Step 5: Perform weighted fitting with error estimation
    try:
        # Create weights based on signal strength and noise level
        signal_weights = 1.0 / (noise_std**2 + 0.1 * np.abs(y_fit_corrected))
        signal_weights = signal_weights / np.max(signal_weights)  # Normalize

        # Fit Gaussian with weights
        popt, _ = curve_fit(
            gaussian,
            x_fit,
            y_fit,
            p0=p0,
            bounds=bounds,
            sigma=1.0 / np.sqrt(signal_weights),
            absolute_sigma=False,
            maxfev=5000,
        )

        amp, mu, sigma, baseline = popt

        # Calculate fitting quality metrics
        y_pred = gaussian(x_fit, *popt)
        residuals = y_fit - y_pred
        chi_squared = np.sum((residuals**2) * signal_weights)
        reduced_chi_squared = chi_squared / max(1, len(x_fit) - 4)

        # Calculate R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        fit_quality = {
            "reduced_chi_squared": reduced_chi_squared,
            "r_squared": r_squared,
            "fit_converged": True,
        }

        # Step 6: Calculate professional SNR using fitted parameters on RAW data
        pulse_width_samples = 2.355 * sigma  # FWHM in samples

        # Define integration region around fitted peak (+-1.177*sigma for FWHM)
        integration_half_width = 1.177 * sigma
        left_idx = int(np.round(mu - integration_half_width))
        right_idx = int(np.round(mu + integration_half_width))

        # Ensure indices are within bounds
        left_idx = max(0, left_idx)
        right_idx = min(n_time - 1, right_idx)

        n_integration_samples = right_idx - left_idx + 1

        if n_integration_samples > 0:
            # Integrate RAW signal over FWHM region
            signal_sum = np.sum(time_series_raw[left_idx : right_idx + 1])
            expected_noise = noise_mean * n_integration_samples

            # SNR calculation with proper error propagation
            snr = (signal_sum - expected_noise) / (noise_std * np.sqrt(n_integration_samples))
        else:
            snr = -1

        peak_idx_fit = int(np.round(mu))

        return snr, pulse_width_samples, peak_idx_fit, (noise_mean, noise_std, fit_quality)

    except Exception as exc:
        print(f"Gaussian fitting failed: {exc}")

        # Fallback: simple peak analysis
        peak_idx_fit = peak_idx_global

        # Estimate width from half-maximum points on RAW data
        half_max = (np.max(y_fit_corrected) + noise_mean) / 2
        above_half_max = y_fit_corrected > (half_max - noise_mean)

        if np.any(above_half_max):
            width_indices = np.where(above_half_max)[0]
            pulse_width_samples = len(width_indices)

            # Map back to global indices for integration
            start_local = width_indices[0]
            end_local = width_indices[-1]
            left_idx = fit_start + start_local
            right_idx = fit_start + end_local

            n_integration_samples = right_idx - left_idx + 1

            # Integrate RAW signal
            signal_sum = np.sum(time_series_raw[left_idx : right_idx + 1])
            expected_noise = noise_mean * n_integration_samples
            snr = (signal_sum - expected_noise) / (noise_std * np.sqrt(n_integration_samples))

        else:
            pulse_width_samples = 1  # Minimum reasonable width for narrow pulse
            # Fallback to peak SNR on RAW data
            signal_peak = np.max(time_series_raw)
            snr = (signal_peak - noise_mean) / noise_std if noise_std > 0 else -1

        fit_quality = {
            "reduced_chi_squared": -1,
            "r_squared": -1,
            "fit_converged": False,
        }

        return snr, pulse_width_samples, peak_idx_fit, (noise_mean, noise_std, fit_quality)


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
