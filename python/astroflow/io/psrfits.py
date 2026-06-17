from astropy.io import fits
import numpy as np
import os

import time
from .. import _astroflow_core as _astro_core  # type: ignore

from .data import SpectrumBase, Header, SpectrumType
from ..config import TaskConfig


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
        self._use_fallback = False  # 标记是否使用回退版本
        self._load_data()
        self._type = SpectrumType.PSRFITS
    
    @iotimeit
    def _load_data(self):
        taskconfig = TaskConfig()
        if taskconfig.psrfitsbackend == "cpp":
            try:
                self._load_data_cpp()
            except Exception as e:
                print(f"[Warning] C++ backend failed with error: {e}. Falling back to Python implementation.")
                self._use_fallback = True
                self._load_data_python()
        else:
            self._use_fallback = True
            self._load_data_python()
        
    def _load_data_cpp(self):
        """使用C++核心库加载PSRFITS文件"""
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
            raj=self._core_instance.raj,
            decj=self._core_instance.decj,
        )

    def _load_data_python(self):
        """Python fallback implementation for loading PSRFITS files using astropy."""
        with fits.open(self.filename) as hdul:  # memmap=True 更稳
            header0 = hdul[0].header  # type: ignore
            header1 = hdul[1].header  # type: ignore
            data = hdul[1].data  # type: ignore

        fch1 = header0["OBSFREQ"] - header0["OBSBW"] / 2
        mjd = header0["STT_IMJD"] + header0["STT_SMJD"] / 86400.0 + header0["STT_OFFS"] / 86400.0
        
        dtype = data["DATA"].dtype
        nchan = header1["NCHAN"]
        data = data["DATA"].reshape(-1, header1["NPOL"], nchan)
        if header1["NPOL"] == 1:  # AABB AA
            data_ = data[:, 0, :]
        elif header1["NPOL"] >= 2:
            data_ = ((data[:, 0, :] + data[:, 1, :]) / 2).astype(dtype)
        else:
            raise ValueError(f"Unsupported NPOL value: {header1['NPOL']}, POL_TYPE: {header1['POL_TYPE']}")
        
        foff = header1["CHAN_BW"]
        nchans = header1["NCHAN"]
        raj = header0.get("RA", header0.get("RAJ"))
        decj = header0.get("DEC", header0.get("DECJ"))

        if foff < 0:
            foff = -foff
            fch1 = fch1 - (nchans - 1) * foff
            data_ = np.flip(data_, axis=1)
        
        self._data = np.ascontiguousarray(data_)
        if len(data_.shape) == 3:
            ndata = data_.shape[0] * data_.shape[1]
        elif len(data_.shape) == 2:
            ndata = data_.shape[0]
        else:
            ndata = data_.shape[0] * data_.shape[1] if len(data_.shape) > 1 else data_.shape[0] // nchans
        
        self._header = Header(
            mjd=mjd,
            filename=self.filename,
            nifs=1,
            nchans=nchans,
            ndata=ndata,
            tsamp=header1["TBIN"],
            fch1=fch1,
            foff=foff,
            nbits=header1["NBITS"],
            raj=raj,
            decj=decj,
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
