import time
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from .io.data import Header, SpectrumBase

def dedtimeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        # print(f"{func.__name__} took {time.time() - start} seconds")
        print(f"[INFO] C++ backend took {(time.time() - start):2f} seconds")
        return result

    return wrapper

def get_freq_end_toa(header: Header, ref_freq: float, ref_freq_toa: float, dm: float) -> float:
    """Convert TOA from reference frequency to header's freq_end frequency.
    
    Args:
        header: Header object containing frequency information
        ref_freq: Reference frequency where ref_freq_toa is measured
        ref_freq_toa: Time of arrival at reference frequency
        dm: Dispersion measure value
        
    Returns:
        Time of arrival at header's freq_end frequency
    """
    DISPERSION_CONSTANT = 4148.808
    fch1 = header.fch1
    foff = header.foff
    nchan = header.nchans
    freq_end = fch1 + foff * (nchan - 1)
    time_latency = DISPERSION_CONSTANT * dm * (1 / (ref_freq**2) - 1 / (freq_end**2))
    return ref_freq_toa - time_latency


class SingleDmConfig:
    def __init__(self, dm, freq_start, freq_end, t_sample):
        self.dm = dm
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.t_sample = t_sample

class Config:
    def __init__(
        self,
        dm_low,
        dm_high,
        freq_start,
        freq_end,
        dm_step,
        time_downsample,
        t_sample,
        confidence=0.5,
    ):
        self.dm_low = dm_low
        self.dm_high = dm_high
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.dm_step = dm_step
        self.time_downsample = time_downsample
        self.t_sample = t_sample
        self.confidence = confidence

    def __str__(self):
        info = f"{self.dm_low}_{self.dm_high}_{self.freq_start}MHz_{self.freq_end}MHz_{self.dm_step}_{self.t_sample}s"
        return info

    def __repr__(self):
        return self.__str__()
    