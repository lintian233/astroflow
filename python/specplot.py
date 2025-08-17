#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astroflow.dedispered import dedisperse_spec_with_dm
from astroflow.io.filterbank import Filterbank
from astroflow.io.psrfits import PsrFits
from astroflow.dataset.generate import get_ref_freq_toa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the dedispersed spectrum plotter.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Plot dedispersed spectrum from radio astronomy data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the filterbank (.fil) or PSRFITS (.fits) file"
    )
    parser.add_argument(
        "toa", 
        type=float, 
        help="Time of arrival (center time) in seconds"
    )
    parser.add_argument(
        "dm", 
        type=float, 
        help="Dispersion measure in pc cm^-3"
    )
    parser.add_argument(
        "output_path", 
        type=str, 
        help="Directory path to save the output plot"
    )
    
    # Optional arguments
    parser.add_argument(
        "--tband", 
        type=float, 
        default=100, 
        help="Time window around TOA in ms"
    )
    parser.add_argument(
        "--freq_start", 
        type=float, 
        default=-1, 
        help="Start frequency in MHz (use -1 for auto)"
    )
    parser.add_argument(
        "--freq_end", 
        type=float, 
        default=-1, 
        help="End frequency in MHz (use -1 for auto)"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to the mask file containing bad channel indices"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Resolution of the output plot in DPI"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10, 10],
        help="Figure size in inches (width height)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()

def load_mask(mask_file: str) -> Optional[np.ndarray]:
    """
    Load channel mask from file.
    
    Args:
        mask_file: Path to the mask file containing bad channel indices
        
    Returns:
        Array of bad channel indices, or None if file doesn't exist
        
    Raises:
        FileNotFoundError: If mask file doesn't exist
        ValueError: If mask file format is invalid
    """
    if not mask_file or not os.path.exists(mask_file):
        return None
        
    try:
        with open(mask_file, 'r') as f:
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
    """
    Load radio astronomy data from file.
    
    Args:
        file_path: Path to the data file (.fil or .fits)
        
    Returns:
        Loaded data object (Filterbank or PsrFits)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext == ".fil":
            return Filterbank(file_path)
        elif file_ext == ".fits":
            return PsrFits(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported: .fil, .fits")
    except Exception as e:
        raise ValueError(f"Error loading data file {file_path}: {e}")


def calculate_dynamic_range(data: np.ndarray, 
                          percentiles: Tuple[float, float] = (0, 100)) -> Tuple[float, float]:
    """
    Calculate dynamic range for data visualization.
    
    Args:
        data: Input data array
        percentiles: Lower and upper percentiles for clipping
        
    Returns:
        Tuple of (vmin, vmax) values for plotting
    """
    vmin, vmax = np.percentile(data, percentiles)
    
    # Handle edge case where vmin is 0
    if vmin == 0:
        non_zero_values = data[data > 0]
        if non_zero_values.size > 0:
            vmin = non_zero_values.min()
        else:
            vmin = data.min()
    
    logger.debug(f"Dynamic range: vmin={vmin:.3e}, vmax={vmax:.3e}")
    return vmin, vmax


def apply_channel_mask(data: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Apply channel mask to data by zeroing out bad channels.
    
    Args:
        data: Input data array (time, frequency)
        mask: Array of bad channel indices
        
    Returns:
        Masked data array
    """
    if mask is None:
        return data.copy()
    
    masked_data = data.copy()
    
    # Validate mask indices
    valid_mask = mask[mask < data.shape[1]]
    if len(valid_mask) != len(mask):
        logger.warning(f"Some mask indices exceed data dimensions, ignoring invalid indices")
    
    if len(valid_mask) > 0:
        masked_data[:, valid_mask] = 0
        logger.info(f"Applied mask to {len(valid_mask)} channels")
    
    return masked_data

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
    figsize: Tuple[float, float] = (10, 10)
) -> str:
    """
    Generate a dedispersed spectrum plot from radio astronomy data.
    
    Args:
        file_path: Path to the input data file
        toa: Time of arrival (center time) in seconds
        dm: Dispersion measure in pc cm^-3
        output_path: Directory to save the output plot
        tband: Time window around TOA in seconds
        freq_start: Start frequency in MHz (-1 for auto)
        freq_end: End frequency in MHz (-1 for auto)
        mask: Array of bad channel indices to mask
        dpi: Plot resolution in DPI
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Path to the saved plot file
        
    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If input file doesn't exist
    """
    # Validate inputs
    if tband <= 0:
        raise ValueError("Time band must be positive")
    if dm < 0:
        raise ValueError("Dispersion measure must be non-negative")

    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    data_obj = load_data(file_path)
    toa = get_ref_freq_toa(data_obj.header(), freq_end, toa, dm)
    # Calculate time window
    tband = tband / 1000  # Convert ms to seconds
    tstart = max(0, toa - tband / 2)
    tend = toa + tband / 2
    
    # Round to avoid floating point precision issues
    tstart = round(tstart, 6)
    tend = round(tend, 6)

    logger.info(f"Processing time window: {tstart:.6f} - {tend:.6f} s (DM={dm:.3f})")    
    # Dedisperse spectrum
    logger.info("Dedispersing spectrum...")
    spectrum = dedisperse_spec_with_dm(
        data_obj, tstart, tend, dm, freq_start, freq_end
    )
    
    header = data_obj.header()
    
    # Apply channel mask if provided
    data = apply_channel_mask(spectrum.data, mask)
    
    # Calculate plot parameters
    vmin, vmax = calculate_dynamic_range(data)
    time_axis = np.linspace(tstart, tend, spectrum.ntimes)
    
    # Calculate frequency axis
    if freq_start == -1:
        freq_start = header.fch1
    freq_axis = freq_start + np.arange(spectrum.nchans) * header.foff
    
    # Calculate integrated time series
    time_series = data.sum(axis=1)
    
    # Create plot
    fig, (ax_time, ax_spec) = plt.subplots(
        2, 1,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True
    )
    
    # Set matplotlib parameters
    plt.rcParams["image.origin"] = "lower"
    
    # Plot integrated time series
    ax_time.plot(time_axis, time_series, "k-", linewidth=0.8, alpha=0.8)
    ax_time.set_ylabel("Integrated Power")
    ax_time.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_time.set_yscale("log")
    ax_time.grid(True, alpha=0.3)
    
    # Create title
    basename = Path(file_path).stem
    title = f"{basename} | t={tstart:.3f}-{tend:.3f}s | DM={dm:.3f} pc cm⁻³"
    ax_time.set_title(title, fontsize=12, fontweight='bold')
    
    # Plot dynamic spectrum
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    im = ax_spec.imshow(
        data.T,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )


    # Set labels with technical details
    ax_spec.set_ylabel(
        f"Frequency (MHz)\n"
        f"f₀={header.fch1:.3f} MHz, Δf={header.foff:.3f} MHz"
    )
    ax_spec.set_xlabel(
        f"Time (s)\n"
        f"Δt={header.tsamp:.3e} s, N_chan={spectrum.nchans}"
    )
    
    # Set axis limits
    ax_time.set_xlim(tstart, tend)
    ax_spec.set_xlim(tstart, tend)
    
    # Improve layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    
    # Generate output filename
    output_filename = f"{basename}_t{tstart:.3f}-{tend:.3f}_DM{dm:.3f}.png"
    output_file = os.path.join(output_path, output_filename)
    
    # Save plot
    plt.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png"
    )
    
    logger.info(f"Saved plot: {output_file}")
    plt.close(fig)
    
    return output_file
def main() -> None:
    """Main function to execute the dedispersed spectrum plotter."""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting dedispersed spectrum plotter")
    logger.info(f"Input file: {args.file_path}")
    logger.info(f"TOA: {args.toa:.6f} s, Time band: {args.tband:.6f} ms")
    logger.info(f"DM: {args.dm:.3f} pc cm^-3")
    logger.info(f"Output directory: {args.output_path}")
    
    try:
        # Load mask if provided
        mask = load_mask(args.mask) if args.mask else None
        # Generate plot
        output_file = plot_dedispersed_spectrum(
            file_path=args.file_path,
            toa=args.toa,
            dm=args.dm,
            output_path=args.output_path,
            tband=args.tband,
            freq_start=args.freq_start,
            freq_end=args.freq_end,
            mask=mask,
            dpi=args.dpi,
            figsize=tuple(args.figsize)
        )
        
        logger.info(f"Successfully generated plot: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        raise


if __name__ == "__main__":
    main()
