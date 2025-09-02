import os
import numpy as np
import cv2

from typing import List

from . import _astroflow_core as _astro_core #type: ignore


from .utils import dedtimeit, Config
from .dmtime import DmTime
from .io.filterbank import Filterbank
from .spectrum import Spectrum
from .io.data import Header, SpectrumBase, SpectrumType
from .config import TaskConfig


@dedtimeit
def dedisperse_spec(
    spectrum: SpectrumBase,
    dm_low: float,
    dm_high: float,
    freq_start: float,
    freq_end: float,
    dm_step: float = 1,
    time_downsample: int = 64,
    t_sample: float = 0.5,
    target: int = 1,
    maskfile="mask.txt",
) -> List[DmTime]:

    if spectrum.type == SpectrumType.FIL:
        filename = spectrum.filename
        dmtimes = dedispered_fil(
            filename,
            dm_low,
            dm_high,
            freq_start,
            freq_end,
            dm_step,
            time_downsample,
            t_sample,
            target,
            maskfile,
        )
        return dmtimes

    spec, header = spectrum.core_data
    data = _astro_core._dedisperse_spec(
        spec,
        header,
        dm_low,
        dm_high,
        freq_start,
        freq_end,
        dm_step,
        time_downsample,
        t_sample,
        TaskConfig().dedgpu,
        maskfile,
        TaskConfig().rficonfig
    )

    basename = os.path.basename(spectrum.filename).split(".")[0]
    result = []
    for idx, dmt in enumerate(data.dm_times):
        tstart = idx * t_sample
        tend = (idx + 1) * t_sample
        result.append(
            DmTime(
                tstart=tstart,
                tend=tend,
                dm_low=dm_low,
                dm_high=dm_high,
                dm_step=dm_step,
                freq_start=freq_start,
                freq_end=freq_end,
                data=dmt,
                name=basename,
            )
        )

    return result


def dedisperse_spec_with_dm(
    spectrum: SpectrumBase,
    tstart: float,
    tend: float,
    dm: float,
    freq_start: float = -1,
    freq_end: float = -1,
    maskfile="mask.txt",
) -> Spectrum:
    if spectrum.type == SpectrumType.FIL:
        fil = Filterbank(spectrum.filename)
        return dedispered_fil_with_dm(fil, tstart, tend, dm, freq_start, freq_end, maskfile)

    spec, header = spectrum.core_data
    if freq_start == freq_end == -1:
        freq_start = header.fch1
        freq_end = header.fch1 + (header.nchans - 1) * header.foff

    data = _astro_core._dedisperse_spec_with_dm(
        spec, header, tstart, tend, dm, freq_start, freq_end, maskfile, TaskConfig().rficonfig
    )

    return Spectrum.from_core_spectrum(data)


def dedispered_fil_with_dm(
    fil: Filterbank,
    tstart: float,
    tend: float,
    dm: float,
    freq_start: float = -1,
    freq_end: float = -1,
    maskfile = "mask.txt",
) -> Spectrum:
    """
    Dedisperse filterbank data at a specific dispersion measure (DM).

    This OpenMP-accelerated function applies inverse dispersion correction to a
    selected time range of filterbank data using a single DM value. The output
    is a frequency-time spectrum where dispersion delays have been removed.

    Parameters
    ----------
    fil : Filterbank
        SIGPROC filterbank data container object.
        Supports 8/16/32-bit quantization.
        Only supoort single polarization data. nifs=1, nbits=8/16/32.

    tstart : float
        Start time (seconds) relative to filterbank start time.
        Must satisfy: 0 ≤ tstart < tend < fil.tsamp * fil.ndata(fil.nsamples)

    tend : float
        End time (seconds) relative to filterbank start time.
        Must satisfy: 0 ≤ tstart < tend < fil.tsamp * fil.ndata(fil.nsamples)

    dm : float
        Dispersion measure (pc/cm³) to apply.
        typical values: 10-1e5 pc/cm³for pulsar observations.

    freq_start : float, optional, default: -1
        Start frequency (MHz) for dedispersion calculation.
        When set to -1, uses fil.fch1 (lowest frequency)
    freq_end : float, optional, default: -1
        End frequency (MHz) for dedispersion calculation.
        When set to -1, uses fil.fch1 + (nchans-1)*foff (highest frequency)


    Returns
    -------
    Spectrum
        Object containing dedispersed data with attributes:
        - name: Base filename identifier
        - data: dedispersed data ndarray: [ntimes, nchans]
        - dm: Dispersion measure applied
        - tstart/tend: Start/end time
        - ntimes: Number of time samples
        - nchans: Number of frequency channels
        - freq_start/end: Actual frequency range used


    Raises
    ------
    ValueError
        - If fil.nbits not in {8, 16, 32}
        - If freq_start >= freq_end (after auto-set)
        - If tstart/tend outside valid time range
        - If DM exceeds GPU memory capacity

    Notes
    -----
    1. **DM Calculation**
       - Uses exact (non-approximated) dispersion delay formula:
         Δt = 4.148741601e3 * DM * (f**(-2) - f_ref**(-2)))
       - f_ref is the highest frequency in selected band

    Examples
    --------
    >>> from astroflow import Filterbank, dedispered_fil_with_dm
    >>> fil = Filterbank("pulsar.fil")
    >>> spec = dedispered_fil_with_dm(fil,
    ...                               tstart=0.5,
    ...                               tend=1.0,
    ...                               dm=56.4)
    >>> print(spec.data.shape)
    (336, 500)  # 336 channels, 0.5s / 1ms tsamp = 500 samples

    # Visualize dedispersed spectrum
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(spec.data.T,
    ...            aspect='auto',
    ...            extent=[spec.fch1, spec.fch1 + spec.foff*spec.data.shape[0],
    ...                    spec.tstart, spec.tend],
    ...            origin='lower')
    >>> plt.colorbar(label="Intensity (arb. units)")
    >>> plt.xlabel("Frequency (MHz)")
    >>> plt.ylabel("Time (s)")

    See Also
    --------
    dedispered_fil : Dedispersion over DM range
    Filterbank : Input data container class
    """
    header = fil.header()
    if freq_start == freq_end == -1:
        freq_start = header.fch1
        freq_end = header.fch1 + (header.nchans - 1) * header.foff

    rficonfig = TaskConfig().rficonfig
    nbits = header.nbits
    if nbits == 8:
        data = _astro_core._dedispered_fil_with_dm_uint8(
            fil.core_instance, tstart, tend, dm, freq_start, freq_end, maskfile, rficonfig
        )
    elif nbits == 16:
        data = _astro_core._dedispered_fil_with_dm_uint16(
            fil.core_instance, tstart, tend, dm, freq_start, freq_end, maskfile, rficonfig
        )
    elif nbits == 32:
        data = _astro_core._dedispered_fil_with_dm_uint32(
            fil.core_instance, tstart, tend, dm, freq_start, freq_end, maskfile, rficonfig
        )
    else:
        raise ValueError(f"Unsupported number of bits: {nbits}")

    return Spectrum.from_core_spectrum(data)


# @timeit
def dedispered_fil(
    file_path: str,
    dm_low: float,
    dm_high: float,
    freq_start: float,
    freq_end: float,
    dm_step: float = 1,
    time_downsample: int = 64,
    t_sample: float = 0.5,
    target: int = 1,
    maskfile = "mask.txt",
) -> List[DmTime]:
    """
    Perform GPU/OpenMP-accelerated dedispersion on filterbank data.

    This function reads a filterbank file, dedisperses the data over a range of
    dispersion measures (DMs), and returns a list of dedispersed data segments.
    Each segment is a 2D array of shape [dm_trials, time_samples] containing
    dedispersed intensity values for each DM trial.

    Parameters
    ----------
    file_path : str
        Path to filterbank file (.fil) in SIGPROC format.
        Supports 8/16/32-bit quantization.
        Only supoort single polarization data. nifs=1, nbits=8/16/32.

    dm_low : float
        Lower bound of dispersion measure (DM) in pc/cm³.
        Must satisfy: 0 ≤ dm_low < dm_high

    dm_high : float
        Upper bound of dispersion measure (DM) in pc/cm³.
        Must satisfy: dm_high > dm_low

    freq_start : float
        Start frequency (MHz) for dedispersion calculation.
        Should match channelization in filterbank header.

    freq_end : float
        End frequency (MHz) for dedispersion calculation.
        Must satisfy: freq_end > freq_start

    dm_step : float, optional, default: 1
        DM trial step size (pc/cm³). Controls search resolution:
        num_dm = floor((dm_high - dm_low) / dm_step) + 1

    time_downsample : int, optional, default: 2
        Temporal downsampling factor.
        Each output sample integrates N=input_samples//downsample samples.

    t_sample : float, optional, default: 0.5
        Effective integration time (seconds) per output sample.

    target : int, optional, default: 1
        Target device for computation:
        - 0: CPU
        - 1: GPU (default)

    Returns
    -------
    List[DmTime]
        Time-ordered list of dedispersed data segments containing:
        - tstart/tend: timestamps (seconds)
        - dm_low/high: Search range parameters
        - dm_step: Trial step size
        - freq_start/end: Actual frequency range used
        - data: 2D ndarray shaped [dm_trials, time_samples]
        - name: Base filename identifier

    Raises
    ------
    ValueError
        For invalid parameter combinations or file inconsistencies
    RuntimeError
        If GPU memory allocation fails or CUDA error occurs.

    Notes
    -----
    1. CPU also supported: Set target=0 in _astro_core._dedispered_fil call.
    2. For large datasets, consider breaking into smaller chunks for memory efficiency.
    3. For optimal performance, use power-of-2 time_downsample values (32, 64, 128, etc.).

    Examples
    --------
    >>> from astroflow import dedispered_fil
    >>> results = dedispered_fil(
    ...     "observation.fil",
    ...     dm_low=100.0,
    ...     dm_high=200.0,
    ...     freq_start=1350.0,
    ...     freq_end=1450.0,
    ...     dm_step=0.5,
    ...     time_downsample=16,
    ...     t_sample=1.0,
    ... )
    >>> # Visualization example
    >>> import matplotlib.pyplot as plt
    >>> dt = results[0]
    >>> plt.imshow(dt.data,
    ...            aspect='auto',
    ...            extent=[dt.tstart, dt.tend, dt.dm_low, dt.dm_high],
    ...            origin='lower')
    >>> plt.xlabel("Time (s)")
    >>> plt.ylabel("DM (pc/cm³)")
    """
    data = _astro_core._dedispered_fil(
        file_path,
        dm_low,
        dm_high,
        freq_start,
        freq_end,
        dm_step,
        time_downsample,
        t_sample,
        target,  # 0 for CPU, 1 for GPU
        TaskConfig().dedgpu,
        maskfile,
        TaskConfig().rficonfig
    )
    basename = os.path.basename(file_path).split(".")[0]
    result = []
    for idx, dmt in enumerate(data.dm_times):
        tstart = idx * t_sample
        tend = (idx + 1) * t_sample
        result.append(
            DmTime(
                tstart=tstart,
                tend=tend,
                dm_low=dm_low,
                dm_high=dm_high,
                dm_step=dm_step,
                freq_start=freq_start,
                freq_end=freq_end,
                data=dmt,
                name=basename,
            )
        )

    return result
