from astropy.io import fits
import numpy as np
import os

import time
from .. import _astroflow_core as _astro_core  # type: ignore

from .data import SpectrumBase, Header, SpectrumType

def iotimeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[INFO] I/O : {(time.time() - start):2f} s")
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
        self._load_data()
        self._type = SpectrumType.PSRFITS
    
    def _load_data(self):
        with fits.open(self.filename) as hdul:  # memmap=True 更稳
            header0 = hdul[0].header
            header1 = hdul[1].header
            data = hdul[1].data

        fch1 = header0["OBSFREQ"] - header0["OBSBW"] / 2
        mjd = header0["STT_IMJD"] + header0["STT_SMJD"] / 86400.0

        data_ = data["DATA"][:, :, 0, :, 0]

        foff = header1["CHAN_BW"]
        nchans = header1["NCHAN"]

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
