#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev

from astroflow.dedispered import dedisperse_spec_with_dm
from astroflow.io.filterbank import Filterbank
from astroflow.io.psrfits import PsrFits


# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ========== CLI ==========
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dedispersed spectrum with baseline removal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file_path", type=str, help="Path to the filterbank (.fil) or PSRFITS (.fits) file")
    parser.add_argument("toa", type=float, help="Time of arrival (center time) in seconds")
    parser.add_argument("dm", type=float, help="Dispersion measure in pc cm^-3")
    parser.add_argument("output_path", type=str, help="Directory path to save the output plot")
    parser.add_argument("--tband", type=float, default=100, help="Time window around TOA in ms")
    parser.add_argument("--freq_start", type=float, default=-1, help="Start frequency in MHz (use -1 for auto)")
    parser.add_argument("--freq_end", type=float, default=-1, help="End frequency in MHz (use -1 for auto)")
    parser.add_argument("--mask", type=str, default=None, help="Path to the mask file containing bad channel indices")
    parser.add_argument("--dpi", type=int, default=100, help="Resolution of the output plot in DPI")
    parser.add_argument("--figsize", type=float, nargs=2, default=[20, 10], help="Figure size in inches (width height)")
    parser.add_argument("--subfreq", type=int, default=None, help="Number of target frequency subbands (default=all channels)")
    parser.add_argument("--subtsamp", type=int, default=1, help="Downsampling factor in time (default=1)")
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )
    return parser.parse_args()


# ========== Helpers ==========
def load_mask(mask_file: str) -> Optional[np.ndarray]:
    if not mask_file or not os.path.exists(mask_file):
        return None
    try:
        with open(mask_file, "r") as f:
            data = f.read().strip()
        if not data:
            logger.warning(f"Mask file {mask_file} is empty")
            return None
        bad_channels = list(map(int, data.split()))
        mask = np.array(bad_channels, dtype=int)
        logger.info(f"Loaded mask with {len(mask)} bad channels from {mask_file}")
        return mask
    except (ValueError, IOError) as e:
        raise ValueError(f"Error reading mask file {mask_file}: {e}")


def load_data(file_path: str) -> Union[Filterbank, PsrFits]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    ext = Path(file_path).suffix.lower()
    if ext == ".fil":
        return Filterbank(file_path)
    elif ext == ".fits":
        return PsrFits(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .fil, .fits")


def calculate_dynamic_range(data: np.ndarray, percentiles=(1, 99.9)) -> Tuple[float, float]:
    finite = np.isfinite(data)
    if not finite.any():
        vmin, vmax = np.percentile(data, percentiles)
    else:
        vmin, vmax = np.percentile(data[finite], percentiles)
    if vmin == vmax:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def apply_channel_mask(data: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return data.copy()
    masked_data = data.copy()
    valid_mask = mask[mask < data.shape[1]]
    if len(valid_mask) != len(mask):
        logger.warning("Some mask indices exceed data dimensions, ignoring invalid")
    if len(valid_mask) > 0:
        masked_data[:, valid_mask] = 0
        logger.info(f"Applied mask to {len(valid_mask)} channels")
    return masked_data


def downsample(data: np.ndarray, subtsamp: int = 1, subfreq: Optional[int] = None) -> np.ndarray:
    """Downsample in time and frequency."""
    ntime, nfreq = data.shape
    # 时间下采样
    if subtsamp > 1:
        new_time = ntime // subtsamp
        data = data[:new_time * subtsamp, :]
        data = data.reshape(new_time, subtsamp, nfreq).mean(axis=1)
    # 频率下采样
    if subfreq is not None and 0 < subfreq < nfreq:
        subband_size = nfreq // subfreq
        data = data[:, :subfreq * subband_size]
        data = data.reshape(data.shape[0], subfreq, subband_size).mean(axis=2)
    return data


def detrend_frequency(data: np.ndarray, poly_order: int = 20) -> np.ndarray:
    """Chebyshev 拟合去除频率方向带通。"""
    ntime, nchan = data.shape
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
            detrended[t, :] = y
    return detrended


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


# ========== Main plotting ==========
def plot_dedispersed_spectrum(
    file_path: str,
    toa: float,
    dm: float,
    output_path: str,
    tband: float = 0.1,
    freq_start: float = -1,
    freq_end: float = -1,
    mask: Optional[np.ndarray] = None,
    dpi: int = 100,
    figsize: Tuple[float, float] = (10, 10),
    subfreq: Optional[int] = None,
    subtsamp: int = 1,
) -> str:
    os.makedirs(output_path, exist_ok=True)

    data_obj = load_data(file_path)
    header = data_obj.header()
    
    # The incoming TOA is at the highest frequency of the observation.
    # fch1 is the lowest frequency.
    f_high = header.fch1 + header.foff * (header.nchans - 1)
    
    # The TOA needs to be converted to the reference frequency for dedispersion,
    # which is conventionally the end frequency of the band of interest (freq_end).
    ref_freq = freq_end
    if ref_freq == -1:
        # If freq_end is not specified, use the lowest frequency of the observation.
        ref_freq = header.fch1 + header.foff * (header.nchans - 1)

    # Dispersion constant in MHz^2 s / (pc cm^-3)
    DISPERSION_CONSTANT = 4.148808e3
    
    # Time delay from f_high to ref_freq
    time_latency = DISPERSION_CONSTANT * dm * (1 / ref_freq**2 - 1 / f_high**2)
    
    # TOA at the reference frequency
    toa_at_ref_freq = toa + time_latency

    tband = tband / 1000
    tstart = max(0, toa_at_ref_freq - tband / 2)
    tend = toa_at_ref_freq + tband / 2

    spectrum = dedisperse_spec_with_dm(data_obj, tstart, tend, dm, freq_start, freq_end)


    data = apply_channel_mask(spectrum.data, mask)

    # 保存原始 bandpass
    bandpass_raw = np.ma.masked_equal(data, 0.0).mean(axis=0).filled(np.nan)

    # 下采样
    data_raw = downsample(data, subtsamp=subtsamp, subfreq=subfreq)

    # 去基线（频率 + 时间）
    data = detrend_frequency(data_raw, poly_order=20)
    data = _detrend(data_raw, axis=0, type='linear')

    # 去基线后的 bandpass
    bandpass_flat = np.ma.masked_equal(data, 0.0).mean(axis=0).filled(np.nan)

    # 动态范围
    vmin, vmax = calculate_dynamic_range(data)
    time_axis = np.linspace(tstart, tend, data.shape[0])
    
    # Resolve freq_start and freq_end if they are set to auto (-1)
    f_start, f_end = freq_start, freq_end
    if f_start == -1:
        f_start = header.fch1
    if f_end == -1:
        f_end = header.fch1 + header.foff * (header.nchans - 1)
    
    freq_axis = np.linspace(f_start, f_end, data.shape[1])

    # === 绘图 ===
    fig, axes = plt.subplots(
        2, 2, figsize=figsize, dpi=dpi,
        gridspec_kw={"height_ratios": [3, 1], "width_ratios": [1, 1]}
    )
    ((ax_spec_raw, ax_spec), (ax_time, ax_bp)) = axes
    plt.rcParams["image.origin"] = "lower"

    basename = Path(file_path).stem
    fig.suptitle(f"{basename} | t={tstart:.3f}-{tend:.3f}s | DM={dm:.3f}", fontsize=14)

    # 原始动态频谱
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    im_raw = ax_spec_raw.imshow(data_raw.T, aspect="auto", cmap="viridis",
                                vmin=np.percentile(data_raw[data_raw != 0], 1),
                                vmax=np.percentile(data_raw[data_raw != 0], 99),
                                extent=extent)
    ax_spec_raw.set_ylabel("Frequency (MHz)")
    ax_spec_raw.set_title("Raw Spectrum")
    ax_spec_raw.tick_params(axis="x", labelbottom=False)

    # 去基线后的动态频谱
    im_detrend = ax_spec.imshow(data.T, aspect="auto", cmap="viridis",
                                vmin=vmin, vmax=vmax, extent=extent)
    ax_spec.set_title("Detrended Spectrum")
    ax_spec.tick_params(axis="y", labelleft=False)
    ax_spec.tick_params(axis="x", labelbottom=False)

    # 时间序列
    time_series = data.sum(axis=1)
    ax_time.plot(time_axis, time_series, "k-", lw=0.8, alpha=0.8)
    ax_time.set_ylabel("Integrated Power")
    ax_time.set_xlabel("Time (s)")
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xlim(extent[0], extent[1])

    # Bandpass 对比
    # ax_bp.plot(freq_axis, bandpass_raw, "r", lw=1.5, label="Raw", alpha=0.7)
    ax_bp.plot(freq_axis, bandpass_flat, "b", lw=1.5, label="Detrended", alpha=0.7)
    ax_bp.set_xlabel("Frequency (MHz)")
    ax_bp.set_ylabel("Mean Amplitude")
    ax_bp.legend()
    ax_bp.grid(True, alpha=0.3)
    ax_bp.set_xlim(freq_axis[0], freq_axis[-1])
    ax_bp.tick_params(axis="y", labelleft=False, labelright=True, right=True)
    ax_bp.yaxis.set_label_position("right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = os.path.join(output_path, f"{basename}_t{tstart:.3f}-{tend:.3f}_DM{dm:.3f}.png")
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved plot: {output_file}")
    return output_file


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    mask = load_mask(args.mask) if args.mask else None
    plot_dedispersed_spectrum(
        args.file_path, args.toa, args.dm, args.output_path,
        tband=args.tband, freq_start=args.freq_start, freq_end=args.freq_end,
        mask=mask, dpi=args.dpi, figsize=tuple(args.figsize),
        subfreq=args.subfreq, subtsamp=args.subtsamp,
    )


if __name__ == "__main__":
    main()
