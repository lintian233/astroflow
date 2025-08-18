from astropy.io import fits
import numpy as np
import os

from .. import _astroflow_core as _astro_core  # type: ignore

from .data import SpectrumBase, Header, SpectrumType


class PsrFits(SpectrumBase):
    """
    Class to handle PSRFITS data files.
    """

    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._load_data()
        self._type = SpectrumType.PSRFITS

    def _load_data(self):
        header0 = None
        header1 = None
        data = None

        with fits.open(self.filename) as hdul:  # type: ignore
            header0 = hdul[0].header
            header1 = hdul[1].header
            data = hdul[1].data

        fch1 = header0["OBSFREQ"] - header0["OBSBW"] / 2
        mjd = header0["STT_IMJD"] + header0["STT_SMJD"] / 86400.0
        data_ = data["DATA"][:, :, 0, :, 0]
        foff = header1["CHAN_BW"]
        nchans = header1["NCHAN"]
        self._data = data_.reshape(-1, data_.shape[2])

        if foff < 0:
            foff = -foff
            fch1 = fch1 - (nchans - 1) * foff
            self._data = np.flip(self._data, axis=1)

        self._header = Header(
            mjd=mjd,
            filename=self.filename,
            nifs=1,
            nchans=header1["NCHAN"],
            ndata=self._data.shape[0],
            tsamp=header1["TBIN"],
            fch1=fch1,
            foff=header1["CHAN_BW"],
            nbits=header1["NBITS"],
        )

    def get_spectrum(self) -> np.ndarray:
        return self._data

    def header(self) -> Header:
        return self._header
