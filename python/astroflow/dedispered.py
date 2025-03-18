import _astroflow_core  as _astro_core # type: ignore

class SearchConfig:
    def __init__(self, dm_low: float, dm_high: float, dm_step: float, freq_start: float, freq_end: float, time_downsample: int, t_sample: float):
        self.dm_low = dm_low
        self.dm_high = dm_high
        self.dm_step = dm_step
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.time_downsample = time_downsample
        self.t_sample = t_sample

def dedispered_fil(
    file_path: str,
    dm_low: float,
    dm_high: float,
    freq_start: float,
    freq_end: float,
    dm_step: float = 1,
    time_downsample: int = 64,
    t_sample: float = 0.5,
) -> _astro_core.DedisperedData:
    """
    Perform dedispersion on filterbank data using uint8 precision with OpenMP parallelization.

    This function implements coherent dedispersion algorithm optimized for radio astronomy data.
    The implementation uses delay-and-add algorithm with SIMD optimizations and parallel processing.

    Parameters
    ----------
    file_path : str
        Path to filterbank file (.fil) containing time-frequency data.
        File should be in SIGPROC filterbank format with 8-bit quantization.

    dm_low : float
        Lower bound of dispersion measure (DM) to search, in pc/cm³.
        Must be non-negative and less than dm_high.

    dm_high : float
        Upper bound of dispersion measure (DM) to search, in pc/cm³.
        Must be greater than dm_low.

    freq_start : float
        Start frequency in MHz for dedispersion calculation.
        Must be within the frequency range of the filterbank file.

    freq_end : float
        End frequency in MHz for dedispersion calculation.
        Must be higher than freq_start and within file's frequency range.

    dm_step : float, optional, default: 1
        Step size for DM trials in pc/cm³.
        Determines the resolution of DM search: (dm_high - dm_low)/dm_step + 1 trials.

    time_downsample : int, optional, default: 64
        Downsampling factor for time axis (applied after dedispersion).
        Each output sample represents sum of `time_downsample` consecutive input samples.

    t_sample : float, optional, default: 0.5
        Integration time per output sample in seconds.
        Must satisfy: t_sample ≈ N * tsamp * time_downsample, where:
        - N is integer number of input samples
        - tsamp is original sampling time from filterbank header

    Returns
    -------
    result : dedisperseddata
        Dedispersed data container with attributes:
        - dm_times : list of ndarray
            Time series for each DM trial, shape (dm_steps, time_samples)
        - shape : tuple
            (number_of_dm_steps, number_of_time_samples)
        - dm_ndata : int
            Number of DM trials (dm_steps)
        - downtsample_ndata : int
            Number of time samples after downsampling
        - dm_low : float
            Copy of input parameter
        - dm_high : float
            Copy of input parameter
        - dm_step : float
            Copy of input parameter
        - tsample : float
            Effective time resolution after downsampling (seconds)
        - filname : str
            Source filename

    Raises
    ------
    ValueError
        If any input parameters are invalid or out of bounds

    Notes
    -----
    1. Dedispersion formula:
        Δt = 4.148808 × 10**3 × DM × (v**-2 - v_ref**-2) [seconds]
        where:
        - DM: Dispersion Measure (pc/cm³)
        - v: Frequency (MHz)
        - v_ref: Reference frequency (highest frequency in band)

    2. Memory requirements scale with:
        O(dm_steps × (time_samples / time_downsample) × nchans)

    3. For optimal performance:
        - Keep working set within CPU cache
        - Use power-of-two downsampling factors
        - Process data in time chunks using t_sample parameter

    Examples
    --------
    >>> from astroflow import dedispered_fil
    >>> result = dedisper_fil_uint8(
    ...     "observation.fil",
    ...     dm_low=100.0,
    ...     dm_high=200.0,
    ...     freq_start=1350.0,
    ...     freq_end=1450.0,
    ...     dm_step=0.5,
    ...     time_downsample=128,
    ...     t_sample=1.0,
    ... )
    >>> print(f"DM trials: {result.dm_ndata}")
    DM trials: 201
    >>> print(f"Time samples: {result.downtsample_ndata}")
    Time samples: 1200
    >>> dm_series = result.dm_times[0]  # First DM trial
    """
    #check gpu availability
    return _astro_core._dedispered_fil(
        file_path,
        dm_low,
        dm_high,
        freq_start,
        freq_end,
        dm_step,
        time_downsample,
        t_sample,
        target=1, # 0 for CPU, 1 for GPU
    )

