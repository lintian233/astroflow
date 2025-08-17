"""
Dataset simulation module for generating synthetic FRB and RFI samples.

This module provides functions to generate synthetic datasets for training
FRB detection models, including both FRB candidates and background RFI samples.
"""

import os
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import cv2
from tqdm import tqdm

from astroflow.io.filterbank import Filterbank, SpectrumType
from astroflow.io.psrfits import PsrFits
from astroflow.dedispered import dedisperse_spec
from astroflow.dataset.generate import get_ref_freq_toa

# Constants
DISPERSION_CONSTANT = 4148.808  # MHz² pc⁻¹ cm³ ms


@dataclass
class SimulationConfig:
    """Configuration for dataset simulation."""
    # Input data
    input_dir: str
    output_dir: str
    
    # Generation parameters
    num_candidates: int = 400
    bg_samples_per_candidate: int = 1
    
    # Signal parameters
    toa_range: Tuple[float, float] = (1, 4)  # seconds
    dm_range: Tuple[float, float] = (320, 570)  # pc cm^-3
    freq_min_range: Tuple[float, float] = (1000, 1300)  # MHz
    freq_max_range: Tuple[float, float] = (1400, 1499)  # MHz
    width_range_ms: Tuple[float, float] = (2, 6)  # milliseconds
    amp_ratio_range: Tuple[float, float] = (0.007, 0.02)
    t_clip_length: float = 0.5  # seconds
    
    # Dedispersion parameters
    dm_low: float = 300
    dm_high: float = 600
    dm_step: float = 0.5
    f_start: float = 1000.0
    f_end: float = 1499.0
    t_down: int = 4
    t_sample: int = 1
    
    # Output image parameters
    image_size: Tuple[int, int] = (512, 512)
    vmin_pct: float = 0
    vmax_pct: float = 100
    
    # Weak FRB criteria
    weak_frb_width_threshold: float = 0.6  # ms
    weak_frb_amp_threshold: float = 0.06
    weak_frb_combined_threshold: float = 1.0  # ms
    
    # Mask file
    maskfile: str = "/home/lingh/work/astroflow/python/none.txt"


def dm_to_delay_samples(dm: float, f_low: float, f_high: float, dt_sample: float) -> int:
    """Convert dispersion measure to delay in samples."""
    delay_s = DISPERSION_CONSTANT * dm * (1.0/f_low**2 - 1.0/f_high**2)
    return int(round(delay_s / dt_sample))


def delay_samples_to_dm(delay_samples: int, f_low: float, f_high: float, dt_sample: float) -> float:
    """Convert delay in samples to dispersion measure."""
    delay_s = delay_samples * dt_sample
    return delay_s / (DISPERSION_CONSTANT * (1.0/f_low**2 - 1.0/f_high**2))


def generate_synthetic_spectrogram(file_path: str, dm: float, toa: float, 
                                 pulse_width_ms: float, pulse_amp_ratio: float,
                                 t_len: float, freq_min: float, freq_max: float):
    """
    Generate a synthetic spectrogram with an injected FRB pulse.
    
    Args:
        file_path: Path to input filterbank or PSRFITS file
        dm: Dispersion measure (pc cm^-3)
        toa: Time of arrival (seconds)
        pulse_width_ms: Pulse width in milliseconds
        pulse_amp_ratio: Pulse amplitude as ratio of max spectrum value
        t_len: Length of time window to extract (seconds)
        freq_min: Minimum frequency for pulse injection (MHz)
        freq_max: Maximum frequency for pulse injection (MHz)
    
    Returns:
        Tuple of (clipped_spectrum, base_object)
    """
    # Load data
    base = Filterbank(file_path) if file_path.endswith('.fil') else PsrFits(file_path)
    hdr = base.core_data[1]
    dt = hdr.tsamp
    n_f = hdr.nchans
    n_t = hdr.ndata
    f_low = hdr.fch1
    f_off = hdr.foff
    f_high = f_low + f_off * (n_f - 1)

    toa_samples = int(toa / dt)
    width_samples = max(1, int(pulse_width_ms / 1000.0 / dt))
    base.settype(SpectrumType.CUSTOM)

    # Get spectrum and prepare for modification
    spec = base.get_spectrum().astype(np.uint16, copy=False)
    max_amp = int(spec.max())
    amp = pulse_amp_ratio * max_amp
    noise_scale = 0.005 * amp

    # Inject pulse across frequency channels (vectorized)
    idx_all = np.arange(n_f, dtype=np.int32)
    freqs = f_low + f_off * idx_all
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    valid_idx = idx_all[mask]
    
    if valid_idx.size > 0:
        inv_fhigh2 = 1.0 / (f_high * f_high)
        inv_f2 = 1.0 / (freqs[valid_idx] * freqs[valid_idx])
        delays = np.rint(DISPERSION_CONSTANT * dm * (inv_f2 - inv_fhigh2) / dt).astype(np.int64)

        for j in range(valid_idx.size):
            i = int(valid_idx[j])
            center = toa_samples + int(delays[j]) + np.random.randint(-10, 10)
            if center <= 0 or center >= n_t:
                continue

            j0 = np.random.randint(-30, 30)
            j1 = np.random.randint(-30, 30)

            t0 = max(0, center - width_samples) + j0
            t1 = min(n_t, center + width_samples) + j1
            if t1 <= t0:
                continue

            t_idx = np.arange(t0, t1, dtype=np.int32)
            g = amp * np.exp(-0.5 * ((t_idx - center) / float(width_samples))**2, dtype=np.float64)
            if noise_scale > 0:
                g += np.random.normal(0.0, noise_scale, size=g.shape)

            g = g.astype(np.uint16, copy=False)
            spec[t0:t1, i] += g

    # Clip and convert spectrum
    np.clip(spec, 0, 255, out=spec)
    spec = spec.astype(np.uint8, copy=False)
    base.spectrumset(spec.ravel())

    # Extract time window around pulse
    t_clip = int(t_len / dt)
    start_t = max(0, toa_samples)
    end_t = start_t + t_clip
    if end_t > n_t:
        end_t = n_t
        start_t = max(0, end_t - t_clip)

    clip_spec = spec[start_t:end_t]
    return clip_spec, base


def dedisperse_get_list(spectrum, dm_low: float, dm_high: float, f_start: float, f_end: float,
                       dm_step: float, t_down: int, t_sample: int, maskfile: str):
    """Get list of DmTime objects from dedispersion."""
    return dedisperse_spec(spectrum, dm_low, dm_high, f_start, f_end, 
                          dm_step, t_down, t_sample, maskfile=maskfile)


def split_dmt_by_toa(dmts, toa: float, min_gap: float = 0.0):
    """
    Split DMT objects into pulse and background pools.
    
    Args:
        dmts: List of DmTime objects
        toa: Time of arrival for the pulse
        min_gap: Minimum time gap for background samples
    
    Returns:
        Tuple of (pulse_dmt, background_pool)
    """
    pulse = None
    bg_pool = []
    
    for d in dmts:
        mid = 0.5 * (d.tstart + d.tend)
        if d.tstart < toa < d.tend:
            pulse = d
        elif abs(mid - toa) >= min_gap:
            bg_pool.append(d)
    
    return pulse, bg_pool


def save_dmt_image(dmt, out_path: str, vmin_pct: float = 0, vmax_pct: float = 100):
    """Save DMT object as image."""
    img = dmt.data
    cv2.imwrite(out_path, img)


def generate_yolo_label(dm: float, toa: float, image_size: Tuple[int, int], 
                       dm_low: float, dm_high: float, t_start: float, t_end: float) -> Tuple[float, ...]:
    """
    Generate YOLO format label for FRB detection.
    
    Args:
        dm: Dispersion measure
        toa: Time of arrival
        image_size: Image dimensions (height, width)
        dm_low: Minimum DM in search range
        dm_high: Maximum DM in search range
        t_start: Start time of the image
        t_end: End time of the image
    
    Returns:
        YOLO format label tuple (class, x_center, y_center, width, height)
    """
    dm_range = dm_high - dm_low
    toa_range = t_end - t_start
    
    dm_pos = int((dm - dm_low) / dm_range * image_size[0])
    toa_pos = int((toa - t_start) / toa_range * image_size[1])
    
    dm_pos = min(max(dm_pos, 0), image_size[0] - 1)
    toa_pos = min(max(toa_pos, 0), image_size[1] - 1)
    
    dm_pos = np.round(dm_pos / image_size[0], 4)
    toa_pos = np.round(toa_pos / image_size[1], 4)
    
    return (0, toa_pos, dm_pos, 0.2, 0.2)


def setup_output_directories(output_dir: str) -> Dict[str, str]:
    """Setup output directory structure."""
    dirs = {
        'frb_images': os.path.join(output_dir, 'frb', 'images'),
        'frb_labels': os.path.join(output_dir, 'frb', 'labels'),
        'rfi_images': os.path.join(output_dir, 'rfi', 'images'),
        'rfi_labels': os.path.join(output_dir, 'rfi', 'labels'),
        'weak_frb_images': os.path.join(output_dir, 'weak_frb', 'images'),
        'weak_frb_labels': os.path.join(output_dir, 'weak_frb', 'labels'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def generate_synthetic_dataset(config: SimulationConfig) -> Dict[str, int]:
    """
    Generate synthetic dataset for FRB detection training.
    
    Args:
        config: Simulation configuration
    
    Returns:
        Dictionary with generation statistics
    """
    # Setup directories
    dirs = setup_output_directories(config.output_dir)
    
    # Get input files
    fil_files = [
        os.path.join(config.input_dir, f) 
        for f in os.listdir(config.input_dir) 
        if f.endswith('.fil') or f.endswith('.fits')
    ]
    
    if not fil_files:
        raise ValueError(f"No .fil or .fits files found in {config.input_dir}")
    
    stats = {
        'total_generated': 0,
        'frb_samples': 0,
        'weak_frb_samples': 0,
        'background_samples': 0,
        'skipped': 0
    }
    
    for i in tqdm(range(config.num_candidates), desc='Generating synthetic data'):
        # Randomize parameters
        file_path = random.choice(fil_files)
        dm = np.random.uniform(*config.dm_range)
        toa = np.random.uniform(*config.toa_range)
        width_ms = np.random.uniform(*config.width_range_ms)
        amp_ratio = np.random.uniform(*config.amp_ratio_range)
        f_min = np.random.uniform(*config.freq_min_range)
        f_max = max(f_min + 60, np.random.uniform(*config.freq_max_range))
        
        # Ensure TOA is within valid range
        toa = min(max(toa, config.t_clip_length/2), 4 - config.t_clip_length/2)
        
        # Generate synthetic spectrogram
        try:
            clip_spec, base = generate_synthetic_spectrogram(
                file_path, dm, toa, width_ms, amp_ratio,
                config.t_clip_length, f_min, f_max
            )
            
            ref_toa = get_ref_freq_toa(base.header(), config.f_end, toa, dm)
            
            # Dedisperse and get DMT objects
            dmts = dedisperse_get_list(
                base, config.dm_low, config.dm_high, config.f_start, config.f_end,
                config.dm_step, config.t_down, config.t_sample, config.maskfile
            )
            
            frb_dmt, bg_pool = split_dmt_by_toa(dmts, ref_toa, min_gap=1)
            
            if frb_dmt is None:
                print(f'[Skip] No DMT containing pulse found (DM={dm:.2f}, TOA={toa:.2f})')
                stats['skipped'] += 1
                continue
            
            # Generate filename tag
            stem = os.path.splitext(os.path.basename(file_path))[0]
            ftag = (f'{stem}_dm_{dm:.3f}_toa_{toa:.3f}_pw_{width_ms:.3f}_'
                   f'pa_{amp_ratio:.3f}_freq_{f_min:.3f}_{f_max:.3f}')
            
            # Determine if this is a weak FRB
            is_weak_frb = ((width_ms <= config.weak_frb_width_threshold) or 
                          (width_ms < config.weak_frb_combined_threshold and 
                           amp_ratio <= config.weak_frb_amp_threshold))
            
            if is_weak_frb:
                # Save as weak FRB sample
                img_path = os.path.join(dirs['weak_frb_images'], f'{ftag}.png')
                lab_path = os.path.join(dirs['weak_frb_labels'], f'{ftag}.txt')
                save_dmt_image(frb_dmt, img_path, config.vmin_pct, config.vmax_pct)
                
                label = generate_yolo_label(dm, ref_toa, config.image_size, 
                                          config.dm_low, config.dm_high,
                                          frb_dmt.tstart, frb_dmt.tend)
                with open(lab_path, 'w') as f:
                    f.write(' '.join(map(str, label)) + '\n')
                
                stats['weak_frb_samples'] += 1
            else:
                # Save as regular FRB sample
                img_path = os.path.join(dirs['frb_images'], f'{ftag}.png')
                lab_path = os.path.join(dirs['frb_labels'], f'{ftag}.txt')
                save_dmt_image(frb_dmt, img_path, config.vmin_pct, config.vmax_pct)
                
                label = generate_yolo_label(dm, ref_toa, config.image_size,
                                          config.dm_low, config.dm_high,
                                          frb_dmt.tstart, frb_dmt.tend)
                with open(lab_path, 'w') as f:
                    f.write(' '.join(map(str, label)) + '\n')
                
                stats['frb_samples'] += 1
            
            # Generate background samples
            if bg_pool:
                pick_n = min(config.bg_samples_per_candidate, len(bg_pool))
                idxs = np.random.choice(len(bg_pool), size=pick_n, replace=False)
                
                for kk, idx in enumerate(idxs):
                    bg_dmt = bg_pool[idx]
                    bg_tag = f'bg_{stem}_{i:06d}_b{kk:02d}'
                    bg_img_path = os.path.join(dirs['rfi_images'], f'{bg_tag}_{dm:.2f}_{toa:.2f}.png')
                    bg_lab_path = os.path.join(dirs['rfi_labels'], f'{bg_tag}_{dm:.2f}_{toa:.2f}.txt')
                    
                    save_dmt_image(bg_dmt, bg_img_path, config.vmin_pct, config.vmax_pct)
                    open(bg_lab_path, 'w').close()  # Empty label file
                    
                    stats['background_samples'] += 1
            
            stats['total_generated'] += 1
            
        except Exception as e:
            print(f'[Error] Failed to process {file_path}: {e}')
            stats['skipped'] += 1
            continue
    
    return stats
