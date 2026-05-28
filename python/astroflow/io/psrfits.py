import numpy as np
import os

import time
from .. import _astroflow_core as _astro_core  # type: ignore

from .data import SpectrumBase, Header, SpectrumType

def iotimeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        filename = getattr(args[0], "filename", None) if args else None
        if filename and os.path.exists(filename) and elapsed > 0:
            size_mb = os.path.getsize(filename) / 1024 / 1024
            print(f"[INFO] I/O : {elapsed:.4f} s, speed: {size_mb / elapsed:.2f} MB/s")
        else:
            print(f"[INFO] I/O : {elapsed:.4f} s")
        return result

    return wrapper

class PsrFits(SpectrumBase):
    """
    Class to handle PSRFITS data files.
    """

    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._reshaped_data = None  # 缓存reshape后的数据
        self._core_instance = None
        self._load_data()
        self._type = SpectrumType.PSRFITS
    
    @iotimeit
    def _load_data(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"PSRFITS file not found: {self.filename}")

        self._core_instance = _astro_core.PsrFits(self.filename)
        self._data = self._core_instance.data
        self._reshaped_data = self._data
        self._header = Header(
            mjd=self._core_instance.mjd,
            filename=self.filename,
            nifs=self._core_instance.nifs,
            nchans=self._core_instance.nchans,
            ndata=self._core_instance.ndata,
            tsamp=self._core_instance.tsamp,
            fch1=self._core_instance.fch1,
            foff=self._core_instance.foff,
            nbits=self._core_instance.nbits,
        )

    def get_spectrum(self) -> np.ndarray:
        if self._reshaped_data is None:
            if len(self._data.shape) == 3:
                self._reshaped_data = self._data.reshape((self._header.ndata, self._header.nchans))
            elif len(self._data.shape) == 2:
                self._reshaped_data = self._data
            else:
                # 一维数组: reshape to (ndata, nchans)
                self._reshaped_data = self._data.reshape((self._header.ndata, self._header.nchans))
        return self._reshaped_data

    def get_original_data(self) -> np.ndarray:
        return self._data

    def header(self) -> Header:
        return self._header
