import multiprocessing
import os
import time
import gc

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
 

from .config.taskconfig import TaskConfig
from .dedispered import dedisperse_spec_with_dm
from .dmtime import DmTime
from .io.filterbank import Filterbank
from .io.psrfits import PsrFits
from .spectrum import Spectrum
from .utils import get_freq_end_toa




def error_tracer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            raise

    return wrapper


class PlotterManager:
    def __init__(self, dmtconfig=None, specconfig=None, max_worker=8):
        self.max_worker = max_worker
        self.pool = multiprocessing.Pool(self.max_worker)
        self.dmtconfig = dmtconfig
        self.speconfig = specconfig
        if self.dmtconfig is None:
            self.dmtconfig = {
                "minpercentile": 5,
                "maxpercentile": 99.9,
            }
            self.speconfig = {
                "minpercentile": 5,
                "maxpercentile": 99,
                "tband": 50,  # 50ms
                "mode": "subband" 
            }
    def pack_background(self, dmt: DmTime, candinfo, save_path, file_path):
        self.pool.apply_async(
            pack_background, args=(dmt, candinfo, save_path, file_path),
        )
    
    def pack_candidate(self, dmt: DmTime, candinfo, save_path, file_path):
        self.pool.apply_async(
            pack_candidate, args=(dmt, candinfo, save_path, file_path),
        )

    def plot_candidate(self, dmt: DmTime, candinfo, save_path, file_path):
        self.pool.apply_async(
            plot_candidate, args=(dmt, candinfo, save_path, file_path, self.dmtconfig, self.speconfig),
        )

    def close(self):
        self.pool.close()
        self.pool.join()

def pack_candidate(dmt, candinfo, save_path, file_path):
    IMAGE_PATH = os.path.join(save_path,"frb","images")
    LABEL_PATH = os.path.join(save_path,"frb","labels")

    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(LABEL_PATH, exist_ok=True)

    dm, toa, freq_start, freq_end, dmt_idx, ref_toa, bbox = _parse_candidate_info(candinfo)
    x,y, w, h = bbox if bbox else (0, 0, 0, 0)
    img = dmt.data
    
    name = f"dm_{dm}_toa_{ref_toa:.3f}_{dmt.__str__()}.png"
    label_name = f"dm_{dm}_toa_{ref_toa:.3f}_{dmt.__str__()}.txt"

    cv2.imwrite(os.path.join(IMAGE_PATH, name), img)

    with open(os.path.join(LABEL_PATH, label_name), "w") as f:
        if len(candinfo) == 7:
            f.write(f"0 {x:.2f} {y:.2f} {w:.2f} {h:.2f} \n")
    # print(f"dm_{dm}_toa_{ref_toa:.3f}_{dmt.__str__()}.png saved to {IMAGE_PATH}")


def pack_background(dmt, candinfo, save_path, file_path):
    IMAGE_PATH = os.path.join(save_path,"bg","images")
    LABEL_PATH = os.path.join(save_path,"bg","labels")

    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(LABEL_PATH, exist_ok=True)

    dm, toa, freq_start, freq_end, dmt_idx, ref_toa, bbox = _parse_candidate_info(candinfo)
    
    img = dmt.data
    name = f"bg_dm_{dm}_toa_{ref_toa:.3f}_{dmt.__str__()}.png"
    label_name = f"bg_dm_{dm}_toa_{ref_toa:.3f}_{dmt.__str__()}.txt"

    cv2.imwrite(os.path.join(IMAGE_PATH, name), img)

    open(os.path.join(LABEL_PATH, label_name), "w").close()  # Create an empty label file

    # print(f"bg_dm_{dm}_toa_{ref_toa:.3f}_{dmt.__str__()}.png saved to {IMAGE_PATH}")


def gaussian(x, amp, mu, sigma, baseline):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + baseline

def calculate_frb_snr(spec, noise_range=None, threshold_sigma=5.0, toa_sample_idx=None, fitting_window_samples=None):
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
    
    # Step 1: Frequency-integrated time series with outlier-resistant summation
    time_series = np.sum(spec, axis=1)  # Sum over frequency axis
    # Apply Gaussian filter for smoothing
    time_series = gaussian_filter(time_series, sigma=1)
    n_time = len(time_series)
    x = np.arange(n_time)
    
    # Step 2: Determine fitting region centered on TOA
    if fitting_window_samples is None:
        # Auto-determine fitting window: 20% of total length or minimum 50 samples
        fitting_window_samples = max(50, int(0.2 * n_time))
    
    if toa_sample_idx is not None:
        # Center fitting window around provided TOA
        fit_start = max(0, toa_sample_idx - fitting_window_samples // 2)
        fit_end = min(n_time, toa_sample_idx + fitting_window_samples // 2)
    else:
        # Use peak-centered window if no TOA provided
        rough_peak_idx = np.argmax(time_series)
        fit_start = max(0, rough_peak_idx - fitting_window_samples // 2)
        fit_end = min(n_time, rough_peak_idx + fitting_window_samples // 2)
    
    fitting_region = slice(fit_start, fit_end)
    x_fit = x[fitting_region]
    y_fit = time_series[fitting_region]
    
    # Step 3: Robust baseline estimation using weighted statistics
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
        noise_data = np.concatenate([time_series[region] for region in noise_regions])
    else:
        # Fallback: use edge regions
        edge_size = max(5, n_time // 10)
        noise_data = np.concatenate([time_series[:edge_size], time_series[-edge_size:]])
    
    # Robust baseline estimation using median and MAD
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
    
    # Step 4: Weighted Gaussian fitting with professional parameter estimation
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
            weighted_var = np.average((x_fit - weighted_mean)**2, weights=weights)
            sigma0 = np.sqrt(weighted_var)
        else:
            sigma0 = fitting_window_samples / 6  # Fallback
    except:
        sigma0 = fitting_window_samples / 6
    
    # Ensure reasonable bounds for sigma
    sigma0 = max(1.0, min(sigma0, fitting_window_samples / 3))
    baseline0 = noise_mean
    
    # Setup fitting parameters and bounds
    p0 = [amp0, mu0, sigma0, baseline0]
    
    # Conservative bounds to prevent overfitting
    sigma_min = 0.5
    sigma_max = min(fitting_window_samples / 2, n_time / 4)
    mu_min = fit_start
    mu_max = fit_end - 1
    
    bounds = (
        [0, mu_min, sigma_min, noise_mean - 3*noise_std],  # lower bounds
        [amp0 * 3, mu_max, sigma_max, noise_mean + 3*noise_std]  # upper bounds
    )
    
    # Step 5: Perform weighted fitting with error estimation
    try:
        # Create weights based on signal strength and noise level
        signal_weights = 1.0 / (noise_std**2 + 0.1 * np.abs(y_fit_corrected))
        signal_weights = signal_weights / np.max(signal_weights)  # Normalize
        
        # Fit Gaussian with weights
        popt, pcov = curve_fit(
            gaussian, x_fit, y_fit, 
            p0=p0, 
            bounds=bounds, 
            sigma=1.0/np.sqrt(signal_weights),
            absolute_sigma=False,
            maxfev=5000
        )
        
        amp, mu, sigma, baseline = popt
        
        # Calculate fitting quality metrics
        y_pred = gaussian(x_fit, *popt)
        residuals = y_fit - y_pred
        chi_squared = np.sum((residuals**2) * signal_weights)
        reduced_chi_squared = chi_squared / max(1, len(x_fit) - 4)
        
        # Calculate R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        fit_quality = {
            'reduced_chi_squared': reduced_chi_squared,
            'r_squared': r_squared,
            'fit_converged': True
        }
        
        # Step 6: Calculate professional SNR using fitted parameters
        pulse_width_samples = 2.355 * sigma  # FWHM in samples
        
        # Define integration region around fitted peak (±1.177*sigma for FWHM)
        integration_half_width = 1.177 * sigma
        left_idx = int(np.round(mu - integration_half_width))
        right_idx = int(np.round(mu + integration_half_width))
        
        # Ensure indices are within bounds
        left_idx = max(0, left_idx)
        right_idx = min(n_time - 1, right_idx)
        
        n_integration_samples = right_idx - left_idx + 1
        
        if n_integration_samples > 0:
            # Integrate signal over FWHM region
            signal_sum = np.sum(time_series[left_idx:right_idx + 1])
            expected_noise = noise_mean * n_integration_samples
            
            # SNR calculation with proper error propagation
            snr = (signal_sum - expected_noise) / (noise_std * np.sqrt(n_integration_samples))
        else:
            snr = -1
        
        peak_idx_fit = int(np.round(mu))
        
        return snr, pulse_width_samples, peak_idx_fit, (noise_mean, noise_std, fit_quality)
        
    except Exception as e:
        print(f"Gaussian fitting failed: {e}")
        
        # Fallback: simple peak analysis
        peak_idx_fit = peak_idx_global
        
        # Estimate width from half-maximum points
        half_max = (np.max(y_fit_corrected) + noise_mean) / 2
        above_half_max = y_fit_corrected > (half_max - noise_mean)
        
        if np.any(above_half_max):
            width_indices = np.where(above_half_max)[0]
            pulse_width_samples = len(width_indices)
        else:
            pulse_width_samples = 3  # Minimum reasonable width
        
        # Simple SNR calculation
        signal_peak = np.max(time_series)
        snr = (signal_peak - noise_mean) / noise_std if noise_std > 0 else -1
        
        fit_quality = {
            'reduced_chi_squared': -1,
            'r_squared': -1,
            'fit_converged': False
        }
        
        return snr, pulse_width_samples, peak_idx_fit, (noise_mean, noise_std, fit_quality)


def _parse_candidate_info(candinfo):
    """Parse candidate information into structured format."""
    if len(candinfo) == 7:
        dm, toa, freq_start, freq_end, dmt_idx, (x, y, w, h), ref_toa = candinfo
        return dm, toa, freq_start, freq_end, dmt_idx, ref_toa, (x, y, w, h)
    elif len(candinfo) == 6:
        dm, toa, freq_start, freq_end, dmt_idx, ref_toa = candinfo
        return dm, toa, freq_start, freq_end, dmt_idx, ref_toa, None
    else:
        dm, toa, freq_start, freq_end, dmt_idx = candinfo
        return dm, toa, freq_start, freq_end, dmt_idx, toa, None


def _load_data_file(file_path: str):
    """Load filterbank or psrfits data file."""
    if file_path.endswith(".fil"):
        return Filterbank(file_path)
    elif file_path.endswith(".fits"):
        return PsrFits(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def _prepare_dm_data(dmt: DmTime):
    """Prepare DM-Time data for plotting."""
    dm_data = np.array(dmt.data, dtype=np.float32)
    dm_data = cv2.cvtColor(dm_data,cv2.COLOR_BGR2GRAY)
    dm_data = cv2.normalize(dm_data, None, 0, 255, cv2.NORM_MINMAX)
    time_axis = np.linspace(dmt.tstart, dmt.tend, dm_data.shape[1])
    dm_axis = np.linspace(dmt.dm_low, dmt.dm_high, dm_data.shape[0])
    
    return dm_data, time_axis, dm_axis


def _calculate_spectrum_time_window(toa: float, pulse_width_samples: float, tsamp: float, tband: float, multiplier: float = 40.0):
    """Calculate spectrum time window around TOA based on pulse width."""
    if pulse_width_samples > 0:
        # Use pulse width to determine window size: 50 × pulse_width
        pulse_width_seconds = pulse_width_samples * tsamp
        time_size = multiplier * pulse_width_seconds / 2  # Half window on each side
    else:
        time_size = (tband * 1e-3) / 2 

    spec_tstart = max(0, toa - time_size)
    spec_tend = toa + time_size
    return np.round(spec_tstart, 3), np.round(spec_tend, 3)


def _setup_dm_plots(fig, gs, dm_data, time_axis, dm_axis, dm_vmin, dm_vmax, dm, toa):
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
    ax_main.set_xlabel("Time (s)", fontsize=12, labelpad=10)
    ax_main.set_ylabel("DM (pc cm$^{-3}$)", fontsize=12, labelpad=10)
    # Add dashed circle around the candidate region instead of straight lines
    time_range = time_axis[-1] - time_axis[0]
    dm_range = dm_axis[-1] - dm_axis[0]
    
    # Calculate circle radius as a fraction of the plot dimensions
    radius_time = time_range * 0.05  # 5% of time range
    radius_dm = dm_range * 0.07      # 5% of DM range
    
    # Create ellipse (circle in data coordinates) around the candidate
    circle = mpatches.Ellipse(
        (toa, dm), 
        width=2*radius_time, 
        height=2*radius_dm,
        fill=False, 
        linestyle='--', 
        linewidth=2, 
        edgecolor='white', 
        alpha=0.7,
        label=f'Candidate: DM={dm:.2f}, TOA={toa:.3f}s'
    )
    ax_main.add_patch(circle)

    # DM marginal plot
    dm_sum = np.max(dm_data, axis=1)
    ax_dm.plot(dm_sum, dm_axis, lw=1.5, color="darkblue")
    ax_dm.tick_params(axis="y", labelleft=False)
    ax_dm.grid(alpha=0.3)
    
    # Time marginal plot
    time_sum = np.max(dm_data, axis=0)
    ax_time.plot(time_axis, time_sum, lw=1.5, color="darkred")
    ax_time.tick_params(axis="x", labelbottom=False)
    ax_time.grid(alpha=0.3)
    ax_time.text(0.02, 0.95, f"DM: {dm:.2f} pc $cm^{{-3}}$ \n TOA: {toa:.3f}s", 
                  transform=ax_time.transAxes, fontsize=10, verticalalignment='top', 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    return ax_time, ax_main, ax_dm


def _setup_spectrum_plots(fig, gs, spec_data, spec_time_axis, spec_freq_axis, 
                          spec_tstart, spec_tend, specconfig, header, 
                          toa=None, dm=None, pulse_width=None, snr=None):
    """Setup standard spectrum subplot components without subband analysis."""
    ax_spec_time = fig.add_subplot(gs[0, 2])
    ax_spec = fig.add_subplot(gs[1, 2], sharex=ax_spec_time)
    ax_spec_freq = fig.add_subplot(gs[1, 3], sharey=ax_spec)
    
    # Time series (sum over frequency axis)
    time_series = np.sum(spec_data, axis=1)
    # vmin = first non-zero 
    ax_spec_time.plot(spec_time_axis, time_series, "-", color="black", linewidth=1)
    
    # Add TOA line if provided
    if toa is not None:
        ax_spec_time.axvline(toa, color='blue', linestyle='--', linewidth=1, 
                           alpha=0.8, label=f'TOA: {toa:.3f}s')
    
    # Add SNR and pulse width info
    if snr is not None and pulse_width is not None:
        pulse_width_ms = pulse_width * header.tsamp * 1000 if pulse_width > 0 else -1
        ax_spec_time.text(0.02, 0.96, f"SNR: {snr:.2f}\nPulse Width: {pulse_width_ms:.2f} ms",
                         transform=ax_spec_time.transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor="white", alpha=0.8))
        
    ax_spec_time.set_ylabel("Integrated Power")
    ax_spec_time.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_spec_time.grid(True, alpha=0.3)
    if toa is not None:
        ax_spec_time.legend(fontsize=9, loc='upper right')
    
    # Frequency marginal (sum over time axis)
    freq_series = np.sum(spec_data, axis=0)
     # vmin = first non-zero 
    vmin = freq_series[freq_series > 0].min()
    vmax = freq_series.max()
    ax_spec_freq.plot(freq_series, spec_freq_axis, "-", color="darkblue", linewidth=1)
    ax_spec_freq.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_spec_freq.grid(True, alpha=0.3)
    ax_spec_freq.set_xlabel("Frequency\nIntegrated Power")
    ax_spec_freq.set_xlim(vmin, vmax * 1.01)

    spec_vmin = np.percentile(spec_data, specconfig.get("minpercentile", 0.1))
    spec_vmax = np.percentile(spec_data, specconfig.get("maxpercentile", 99.9))
    if spec_vmin == 0:
        non_zero_values = spec_data[spec_data > 1]
        if non_zero_values.size > 0:
            spec_vmin = non_zero_values.min()
    
    # Main spectrum plot
    extent = [spec_time_axis[0], spec_time_axis[-1], spec_freq_axis[0], spec_freq_axis[-1]]
    im = ax_spec.imshow(
        spec_data.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=extent,
        vmin=spec_vmin,
        vmax=spec_vmax,
    )
    
    # Add TOA line to main spectrum plot
    # if toa is not None:
    #     ax_spec.axvline(toa, color='white', linestyle='--', linewidth=1, 
    #                    alpha=0.8)
    
    ax_spec.set_ylabel(f"Frequency (MHz)\nFCH1={header.fch1:.3f} MHz, FOFF={header.foff:.3f} MHz")
    ax_spec.set_xlabel(f"Time (s)\nTSAMP={header.tsamp:.6e}s")
    ax_spec.set_xlim(spec_tstart, spec_tend)

    return ax_spec_time, ax_spec, ax_spec_freq




def _detrend(data: np.ndarray, axis: int = -1, 
             type: str = 'linear', bp=0, overwrite_data: bool = False) -> np.ndarray:
    """Remove linear or constant trend along axis from data.
    
    Simplified version of scipy.signal.detrend for spectrum data preprocessing.
    
    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`. This parameter
        only has an effect when ``type == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.
    """
    from scipy import linalg
    
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")

    data = np.asarray(data)
    dtype = data.dtype.char
    if dtype not in 'dfDF':
        dtype = 'd'
        
    if type in ['constant', 'c']:
        ret = data - np.mean(data, axis, keepdims=True)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = np.asarray(bp)
        bp = np.sort(np.unique(np.concatenate(np.atleast_1d(0, bp, N))))
        if np.any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")

        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdata = np.moveaxis(data, axis, 0)
        newdata_shape = newdata.shape
        newdata = newdata.reshape(N, -1)

        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)

        # Find leastsq fit and remove it for each piece
        for m in range(len(bp) - 1):
            Npts = bp[m + 1] - bp[m]
            A = np.ones((Npts, 2), dtype)
            A[:, 0] = np.arange(1, Npts + 1, dtype=dtype) / Npts
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - A @ coef

        # Put data back in original shape.
        newdata = newdata.reshape(newdata_shape)
        ret = np.moveaxis(newdata, 0, axis)
        return ret


def _setup_detrend_spectrum_plots(fig, gs, spec_data, spec_time_axis, spec_freq_axis, 
                                  spec_tstart, spec_tend, specconfig, header, 
                                  toa=None, dm=None, pulse_width=None, snr=None,
                                  detrend_type='linear'):
    """Setup spectrum subplot components with detrending applied for better signal visibility.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to add subplots to
    gs : matplotlib.gridspec.GridSpec
        The grid specification for subplot layout
    spec_data : numpy.ndarray
        The spectrum data - expected shape (time, frequency) for consistency with other functions
    spec_time_axis : numpy.ndarray
        Time axis values
    spec_freq_axis : numpy.ndarray
        Frequency axis values
    spec_tstart : float
        Start time for spectrum window
    spec_tend : float
        End time for spectrum window
    specconfig : dict
        Configuration dictionary for spectrum plotting
    header : object
        Data file header with timing/frequency information
    toa : float, optional
        Time of arrival
    dm : float, optional
        Dispersion measure
    pulse_width : float, optional
        Pulse width in samples
    snr : float, optional
        Signal-to-noise ratio
    detrend_type : str, optional
        Type of detrending ('linear' or 'constant'), default 'linear'
        Applied to each frequency channel along the time axis
        
    Returns
    -------
    tuple
        (ax_spec_time, ax_spec, ax_spec_freq) - the three subplot axes
        
    Notes
    -----
    This function performs detrending on each frequency channel along the time axis.
    The input data format is automatically detected and converted as needed.
    Detrending is always applied in (frequency, time) format for optimal results.
    """
    ax_spec_time = fig.add_subplot(gs[0, 2])
    ax_spec = fig.add_subplot(gs[1, 2], sharex=ax_spec_time)
    ax_spec_freq = fig.add_subplot(gs[1, 3], sharey=ax_spec)
    
    # Check if we need to transpose based on axis lengths matching data dimensions
    # If spec_time_axis matches spec_data.shape[1] and spec_freq_axis matches spec_data.shape[0],
    # then spec_data is in (frequency, time) format 
    data_is_freq_time = (len(spec_freq_axis) == spec_data.shape[0] and 
                        len(spec_time_axis) == spec_data.shape[1])
    
    # For detrending, we need data in (frequency, time) format to detrend each frequency channel along time
    if data_is_freq_time:
        # print(f"Data is in (frequency, time) format - perfect for detrending")
        detrend_data = spec_data  # Keep original format for detrending
        display_data = spec_data.T  # Transpose for display (time, frequency)
        # print(f"Detrend data shape: {detrend_data.shape} (freq, time)")
        # print(f"Display data shape: {display_data.shape} (time, freq)")
    else:
        # print(f"Data is in (time, frequency) format - transposing for detrending")
        detrend_data = spec_data.T  # Transpose to (frequency, time) for detrending
        display_data = spec_data  # Keep original for display
        # print(f"Detrend data shape: {detrend_data.shape} (freq, time)")
        # print(f"Display data shape: {display_data.shape} (time, freq)")

    # Apply detrending to the spectrum data in (frequency, time) format
    # This detrends each frequency channel along the time axis (axis=1)
    # print(f"Applying {detrend_type} detrending along time axis (axis=1) for each frequency channel")
    
    try:
        detrended_freq_time = _detrend(detrend_data, axis=1, type=detrend_type)  # Always detrend along time axis
        # print(f"Detrending successful. Original range: [{np.min(detrend_data):.3f}, {np.max(detrend_data):.3f}], "
        #       f"Detrended range: [{np.min(detrended_freq_time):.3f}, {np.max(detrended_freq_time):.3f}]")
        
        # Convert detrended data back to (time, frequency) format for plotting
        detrended_data = detrended_freq_time.T
        # print(f"Detrended data for plotting shape: {detrended_data.shape} (time, freq)")
    except Exception as e:
        print(f"Detrending failed: {e}, using original data")
        detrended_data = display_data

    toa_sample_idx = int((toa - spec_tstart) / header.tsamp)

    snr, pulse_width, peak_idx, (noise_mean, noise_std, fit_quality) = calculate_frb_snr(
                detrended_data, noise_range=None, threshold_sigma=5, toa_sample_idx=toa_sample_idx
    )

    toa = spec_tstart + (peak_idx + 0.5) * header.tsamp  # Convert peak index back to time

    # Time series (sum over frequency axis) - use detrended data
    time_series = np.sum(detrended_data, axis=1)
    # print(f"Time series length: {len(time_series)}, spec_time_axis length: {len(spec_time_axis)}")
    ax_spec_time.plot(spec_time_axis, time_series, "-", color="black", linewidth=1)
    
    # Add TOA line if provided
    if toa is not None:
        ax_spec_time.axvline(toa, color='blue', linestyle='--', linewidth=1, 
                           alpha=0.8, label=f'TOA: {toa:.3f}s')
    
    # Add SNR and pulse width info with detrending information
    info_text = f"SNR: {snr:.2f}" if snr is not None else "SNR: N/A"
    if pulse_width is not None:
        pulse_width_ms = pulse_width * header.tsamp * 1000 if pulse_width > 0 else -1
        info_text += f"\nPulse Width: {pulse_width_ms:.2f} ms"
    info_text += f"\nDetrend: {detrend_type} (per freq channel)"
    
    ax_spec_time.text(0.02, 0.96, info_text,
                     transform=ax_spec_time.transAxes, fontsize=10, 
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                     facecolor="lightyellow", alpha=0.8))
    
    ax_spec_time.set_ylabel("Integrated Power\n(Detrended)")
    ax_spec_time.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_spec_time.grid(True, alpha=0.3)
    if toa is not None:
        ax_spec_time.legend(fontsize=9, loc='upper right')
    
    # Frequency marginal (sum over time axis) - use detrended data
    freq_series = np.sum(detrended_data, axis=0)
    # print(f"Freq series length: {len(freq_series)}, spec_freq_axis length: {len(spec_freq_axis)}")
    ax_spec_freq.plot(freq_series, spec_freq_axis, "-", color="darkblue", linewidth=1)
    ax_spec_freq.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_spec_freq.grid(True, alpha=0.3)
    ax_spec_freq.set_xlabel("Frequency\nIntegrated Power\n(Detrended)")
    
    # Calculate percentiles for detrended data
    spec_vmin = np.percentile(detrended_data, specconfig.get("minpercentile", 0.1))
    spec_vmax = np.percentile(detrended_data, specconfig.get("maxpercentile", 99.9))
    if spec_vmin == 0:
        non_zero_values = detrended_data[detrended_data > 0]
        if non_zero_values.size > 0:
            spec_vmin = non_zero_values.min()
    
    # Main spectrum plot with detrended data
    extent = [spec_time_axis[0], spec_time_axis[-1], spec_freq_axis[0], spec_freq_axis[-1]]
    im = ax_spec.imshow(
        detrended_data.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=extent,
        vmin=spec_vmin,
        vmax=spec_vmax,
    )
    
    ax_spec.set_ylabel(f"Frequency (MHz) - {detrend_type.title()} Detrended\n"
                      f"FCH1={header.fch1:.3f} MHz, FOFF={header.foff:.3f} MHz")
    ax_spec.set_xlabel(f"Time (s)\nTSAMP={header.tsamp:.6e}s")
    ax_spec.set_xlim(spec_tstart, spec_tend)
    
    return ax_spec_time, ax_spec, ax_spec_freq


def _setup_subband_spectrum_plots(fig, gs, spec_data, spec_time_axis, spec_freq_axis, 
                                 spec_tstart, spec_tend, specconfig, header, 
                                 toa=None, dm=None, pulse_width=None, snr=None):
    """Setup spectrum subplot components with subband analysis for enhanced weak pulse visibility."""
    ax_spec_time = fig.add_subplot(gs[0, 2])
    ax_spec = fig.add_subplot(gs[1, 2], sharex=ax_spec_time)
    ax_spec_freq = fig.add_subplot(gs[1, 3], sharey=ax_spec)
    
    time_bin_duration = (pulse_width / 2) * header.tsamp if pulse_width else 4 * header.tsamp
    time_bin_size = max(1, int(time_bin_duration / header.tsamp))  # Convert to samples
    
    n_time_samples, n_freq_channels = spec_data.shape
    
    n_freq_subbands = 128
    freq_subband_size = max(1, n_freq_channels // n_freq_subbands)
    
    n_time_bins = max(1, n_time_samples // time_bin_size)
    
    # print(f"Subband analysis: {n_freq_subbands} freq subbands × {n_time_bins} time bins")
    # print(f"Freq subband size: {freq_subband_size} channels, Time bin size: {time_bin_size} samples ({time_bin_duration*1000:.3f} ms)")
    
    # Step 3: Create subband matrix
    subband_matrix = np.zeros((n_time_bins, n_freq_subbands))
    
    for t_bin in range(n_time_bins):
        t_start = t_bin * time_bin_size
        t_end = min((t_bin + 1) * time_bin_size, n_time_samples)
        
        for f_bin in range(n_freq_subbands):
            f_start = f_bin * freq_subband_size
            f_end = min((f_bin + 1) * freq_subband_size, n_freq_channels)
            
            # Sum all values in this subband cell
            subband_value = np.sum(spec_data[t_start:t_end, f_start:f_end])
            subband_matrix[t_bin, f_bin] = subband_value
    
    for f_bin in range(n_freq_subbands):
        freq_column = subband_matrix[:, f_bin]
        
        # Robust normalization: handle constant or near-constant columns
        col_min = np.min(freq_column)
        col_max = np.max(freq_column)
        denom = col_max - col_min
        if np.isclose(denom, 0) or denom < 1e-10:
            freq_column_norm = np.zeros_like(freq_column)
        else:
            freq_column_norm = (freq_column - col_min) / denom
        subband_matrix[:, f_bin] = freq_column_norm
    


    # Step 4: Create axes for subband visualization
    subband_time_axis = np.linspace(spec_tstart, spec_tend, n_time_bins + 1)
    subband_freq_axis = np.linspace(spec_freq_axis[0], spec_freq_axis[-1], n_freq_subbands + 1)
    
    # Subband time series (sum over frequency subbands)
    subband_time_series = np.sum(subband_matrix, axis=1)
    
    # Plot only subband time series
    subband_time_centers = 0.5 * (subband_time_axis[:-1] + subband_time_axis[1:])
    ax_spec_time.plot(subband_time_centers, subband_time_series, "-", color="black", 
                     linewidth=1, alpha=0.9)
    
    ax_spec_time.text(0.02, 0.96, f"SNR: {snr:.2f} \n"
                      f"pulse width: {pulse_width * header.tsamp * 1000:.2f} ms",
                     transform=ax_spec_time.transAxes, fontsize=10, 
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                     facecolor="white", alpha=0.8))
    
    # Add TOA line if provided
    if toa is not None:
        ax_spec_time.axvline(toa, color='blue', linestyle='--', linewidth=1, 
                           alpha=0.8, label=f'TOA: {toa:.3f}s')
    
    ax_spec_time.set_ylabel("Integrated Power")
    ax_spec_time.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_spec_time.grid(True, alpha=0.3)
    ax_spec_time.legend(fontsize=9, loc='upper right')
    
    # Step 6: Frequency marginal from subband analysis only
    subband_freq_series = np.sum(subband_matrix, axis=0)
    
    # Create frequency centers for subband plotting
    subband_freq_centers = 0.5 * (subband_freq_axis[:-1] + subband_freq_axis[1:])
    
    ax_spec_freq.plot(subband_freq_series, subband_freq_centers, "-", color="black", linewidth=1, 
                     alpha=0.8)
    
    ax_spec_freq.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_spec_freq.grid(True, alpha=0.3)
    ax_spec_freq.set_xlabel("Frequency\nIntegrated Power")
    
    # Main subband spectrum plot
    extent_subband = [subband_time_axis[0], subband_time_axis[-1], 
                     subband_freq_axis[0], subband_freq_axis[-1]]
    
    spec_vmin = np.percentile(subband_matrix, specconfig.get("minpercentile", 0.1))
    spec_vmax = np.percentile(subband_matrix, specconfig.get("maxpercentile", 99.9))

    if spec_vmin == 0:
        non_zero_values = subband_matrix[subband_matrix > 0]
        if non_zero_values.size > 0:
            spec_vmin = non_zero_values.min()

    im = ax_spec.imshow(
        subband_matrix.T,
        aspect="auto",
        origin="lower", 
        cmap="viridis",
        extent=extent_subband,
        vmin=spec_vmin,
        vmax=spec_vmax,
        interpolation='nearest'  # Use nearest neighbor to preserve subband structure
    )
    
    # Add TOA line to main spectrum plot
    # if toa is not None:
    #     ax_spec.axvline(toa, color='white', linestyle='--', linewidth=0.8, 
    #                    alpha=0.8, label='TOA')
    
    # Add grid to show subband boundaries
    # for i in range(1, n_freq_subbands):
    #     freq_boundary = subband_freq_axis[i]
    #     ax_spec.axhline(freq_boundary, color='white', linestyle='-', 
    #                    linewidth=0.5, alpha=0.3)
    
    # for i in range(1, n_time_bins):
    #     time_boundary = subband_time_axis[i]
    #     ax_spec.axvline(time_boundary, color='white', linestyle='-', 
    #                    linewidth=0.3, alpha=0.2)
    
    ax_spec.set_ylabel(f"Frequency (MHz) - {n_freq_subbands} Subbands ({freq_subband_size} channels each)\n"
                      f"FCH1={header.fch1:.3f} MHz, FOFF={header.foff:.3f} MHz")
    ax_spec.set_xlabel(f"Time (s) - {n_time_bins} Bins ({time_bin_duration*1000:.3f} ms each)\n"
                      f"TSAMP={header.tsamp:.6e}s, Bin Size={time_bin_size} samples duration={n_time_bins * time_bin_duration*1000:.1f} ms")
    ax_spec.set_xlim(spec_tstart, spec_tend)
    
    return ax_spec_time, ax_spec, ax_spec_freq


def plot_candidate(
    dmt: DmTime, 
    candinfo, 
    save_path: str, 
    file_path: str, 
    dmtconfig: dict, 
    specconfig: dict, 
    dpi: int = 150
):
    """
    Plot FRB candidate with DM-Time and spectrum analysis.
    
    Creates a comprehensive plot showing both DM-Time data and dedispersed spectrum
    for a Fast Radio Burst candidate, including SNR calculation and pulse analysis.
    
    Args:
        dmt: DmTime object containing dispersion measure vs time data
        candinfo: Candidate information tuple (dm, toa, freq_start, freq_end, dmt_idx, ...)
        save_path: Directory path to save the output plot
        file_path: Path to the original data file (.fil or .fits)
        dmtconfig: Configuration dict for DM-Time plot (minpercentile, maxpercentile)
        specconfig: Configuration dict for spectrum plot (minpercentile, maxpercentile, tband)
        dpi: Resolution for the output plot (default: 150)
    
    Raises:
        ValueError: If file_path has unsupported extension
        Exception: If data loading or processing fails
    """
    try:
        # Parse candidate information
        dm, toa, freq_start, freq_end, dmt_idx, ref_toa, bbox = _parse_candidate_info(candinfo)
        
        print(f"Plot cand: DM={dm}, TOA={toa}, Freq={freq_start}-{freq_end} MHz, DMT Index={dmt_idx}")
        
        # Setup figure and grid
        fig = plt.figure(figsize=(20, 10), dpi=dpi)
        gs = GridSpec(
            2, 4,
            figure=fig,
            width_ratios=[3, 1, 3, 1],
            height_ratios=[1, 3],
            wspace=0.04,
            hspace=0.04,
        )
        
        # Prepare DM-Time data
        dm_data, time_axis, dm_axis = _prepare_dm_data(dmt)
        dm_vmin, dm_vmax = np.percentile(
            dm_data, 
            [dmtconfig.get("minpercentile", 5), dmtconfig.get("maxpercentile", 99.9)]
        )
        
        ax_time, ax_main, ax_dm = _setup_dm_plots(fig, gs, dm_data, time_axis, dm_axis, dm_vmin, dm_vmax, dm, toa)
        
        # Load and process spectrum data
        origin_data = None
        try:
            origin_data = _load_data_file(file_path)
            header = origin_data.header()
            ref_toa = get_freq_end_toa(origin_data.header(), freq_end, toa, dm)
            # First pass: get initial spectrum for pulse width estimation
            tband = specconfig.get("tband", 0.5)  # Default to 0.5 seconds if not specified
            initial_spec_tstart, initial_spec_tend = _calculate_spectrum_time_window(toa, 0, header.tsamp, tband)
            
            taskconfig = TaskConfig()
            basename = os.path.basename(file_path).split(".")[0]
            mask_file_dir = taskconfig.maskdir
            maskfile = f"{mask_file_dir}/{basename}_your_rfi_mask.bad_chans"
    
            if not os.path.exists(maskfile):
                maskfile = taskconfig.maskfile
            # Generate initial dedispersed spectrum for SNR calculation
            initial_spectrum = dedisperse_spec_with_dm(
                origin_data, initial_spec_tstart, initial_spec_tend, dm, freq_start, freq_end, maskfile
            )
            initial_spec_data = initial_spectrum.data
            
            # Calculate SNR and pulse characteristics with TOA-centered fitting
            toa_sample_idx = int((toa - initial_spec_tstart) / header.tsamp)
            toa_sample_idx = max(0, min(toa_sample_idx, initial_spectrum.ntimes - 1))
            
            snr, pulse_width, peak_idx, (noise_mean, noise_std, fit_quality) = calculate_frb_snr(
                initial_spec_data, noise_range=None, threshold_sigma=5, toa_sample_idx=toa_sample_idx
            )

            peak_time = initial_spec_tstart + (peak_idx + 0.5) * header.tsamp
            # Now calculate proper spectrum window based on pulse width (50 × pulse_width)

            spec_tstart, spec_tend = _calculate_spectrum_time_window(peak_time, pulse_width, header.tsamp, tband, 35)
            
            # Generate final spectrum with optimized window
            spectrum = dedisperse_spec_with_dm(
                origin_data, spec_tstart, spec_tend, dm, freq_start, freq_end, maskfile
            )

            spec_data = spectrum.data

            snr, _, peak_idx, (noise_mean, noise_std, fit_quality) = calculate_frb_snr(
                spec_data, noise_range=None, threshold_sigma=5, toa_sample_idx=int((peak_time - spec_tstart) / header.tsamp)
            )

            # Check SNR against threshold
            snrhold = taskconfig.snrhold
            if snr < snrhold:
                plt.close('all')
                if origin_data is not None:
                    if hasattr(origin_data, "close"):
                        try:
                            origin_data.close()
                        except Exception:
                            pass
                    del origin_data
                
                del spectrum, spec_data, initial_spectrum, initial_spec_data
                gc.collect()
                return  

            peak_time = spec_tstart + (peak_idx + 0.5) * header.tsamp
            pulse_width_ms = pulse_width * header.tsamp * 1e3 if pulse_width > 0 else -1  # Convert to milliseconds
            
            # Create time and frequency axes
            spec_time_axis = np.linspace(spec_tstart, spec_tend, spectrum.ntimes)
            
            # Fix frequency axis calculation - use proper frequency mapping
            spec_freq_axis = np.linspace(freq_start, freq_end, spectrum.nchans)
            
            # Setup subband spectrum plots (replaces _setup_spectrum_plots)
            if specconfig.get("mode") == "subband":
                _setup_subband_spectrum_plots(
                    fig, gs, spec_data, spec_time_axis, spec_freq_axis,
                    spec_tstart, spec_tend, specconfig, header, toa=peak_time, dm=dm, pulse_width=pulse_width, snr=snr
                )
            elif specconfig.get("mode") == "standard" or specconfig.get("mode") is None:
                _setup_spectrum_plots(
                    fig, gs, spec_data, spec_time_axis, spec_freq_axis,
                    spec_tstart, spec_tend, specconfig, header, toa=peak_time, dm=dm, pulse_width=pulse_width, snr=snr
                )
            elif specconfig.get("mode") == "detrend":
                _setup_detrend_spectrum_plots(
                    fig, gs, spec_data, spec_time_axis, spec_freq_axis,
                    spec_tstart, spec_tend, specconfig, header, toa=peak_time, dm=dm, pulse_width=pulse_width, snr=snr
                )
            else:
                raise ValueError(f"Unsupported spectrum mode: {specconfig.get('mode')}")
        except Exception as e:
            print(f"Warning: Failed to process spectrum data: {e}")
            # Set default values if spectrum processing fails
            snr, pulse_width_ms, peak_time = -1, -1, toa
        
        # Create title and save plot
        basename = os.path.basename(file_path).split(".")[0]
        fig.suptitle(
            f"FILE: {basename} - DM: {dm} - TOA: {ref_toa:.3f}s - SNR: {snr:.2f} - "
            f"Pulse Width: {pulse_width_ms:.2f} ms - Peak Time: {peak_time:.3f}s",
            fontsize=16,
            y=0.96,
        )
        
        # Generate output filename and save
        output_filename = (
            f"{save_path}/{snr:.2f}_{pulse_width_ms:.2f}_{dm}_{ref_toa:.3f}_{dmt.__str__()}.png"
        )
        print(f"Saving: {os.path.basename(output_filename)}")

        plt.savefig(
            output_filename,
            dpi=dpi,
            format="png",
            bbox_inches="tight",
        )
        
    except Exception as e:
        print(f"Error in plot_candidate: {e}")
        raise
    
    finally:
        # Cleanup
        plt.close('all')
        if 'origin_data' in locals() and origin_data is not None:
            if hasattr(origin_data, "close"):
                try:
                    origin_data.close()
                except Exception:
                    pass
            del origin_data
        gc.collect()
