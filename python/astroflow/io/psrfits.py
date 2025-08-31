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
        self._load_data()
        self._type = SpectrumType.PSRFITS
    
    # @iotimeit
    def _load_data(self):
        with fits.open(self.filename, memmap=True) as hdul:  # memmap=True 更稳
            header0 = hdul[0].header
            header1 = hdul[1].header
            data = hdul[1].data

        fch1 = header0["OBSFREQ"] - header0["OBSBW"] / 2
        mjd = header0["STT_IMJD"] + header0["STT_SMJD"] / 86400.0

        data_ = data["DATA"][:, :, 0, :, 0]

        # 频率步长与通道数
        foff = header1["CHAN_BW"]
        nchans = header1["NCHAN"]

        if foff < 0:
            foff = -foff
            fch1 = fch1 - (nchans - 1) * foff
            data_ = np.flip(data_, axis=1) 

        # data_ = np.ascontiguousarray(data_)

        try:
            self._data = data_.reshape(-1)
        except Exception as e:
            # reval
            self._data = data_.ravel()

        ndata = data_.shape[0] * data_.shape[1]
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
            self._reshaped_data = self._data.reshape((self._header.ndata, self._header.nchans))
        return self._reshaped_data

    def get_original_data(self) -> np.ndarray:
        return self._data

    def header(self) -> Header:
        return self._header
