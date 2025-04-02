# type: ignore
import _astroflow_core as _astro_core
import os

from .data import Header, SpectrumBase
from .data import SpectrumType


class Filterbank(SpectrumBase):
    def __init__(self, filename: str = None):
        super().__init__()
        self._core_instance = None
        if filename is None:
            self.core_instance = _astro_core.Filterbank()

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        if os.path.splitext(filename)[1] not in [".fil", ".FIL"]:
            raise ValueError(f"Invalid file extension: {filename}")

        self._type = SpectrumType
        self._filename = filename
        self._nchans = None
        self._nifs = None
        self._nbits = None
        self._fch1 = None
        self._foff = None
        self._tstart = None
        self._tsamp = None
        self._ndata = None
        self._data = None
        self._header = None

    def _load_data(self):
        self._core_instance = _astro_core.Filterbank(self._filename)
        self._filename = self.core_instance.filename
        self._nchans = self.core_instance.nchans
        self._nifs = self.core_instance.nifs
        self._nbits = self.core_instance.nbits
        self._fch1 = self.core_instance.fch1
        self._foff = self.core_instance.foff
        self._tstart = self.core_instance.tstart
        self._tsamp = self.core_instance.tsamp
        self._ndata = self.core_instance.ndata
        self._data = self.core_instance.data

    @property
    def core_instance(self):
        if self._core_instance is None:
            self._lodad_data()
        return self._core_instance

    def get_spectrum(self):
        if self._data is None:
            self._load_data()
        return self._data

    def header(self) -> Header:
        """
        Returns the header information of the filterbank file.
        """
        if self._data is None:
            self._load_data()

        if self._header is None:
            self._header = Header(
                mjd=self._tstart,
                filename=self._filename,
                nifs=self._nifs,
                nchans=self._nchans,
                ndata=self._ndata,
                tsamp=self._tsamp,
                fch1=self._fch1,
                foff=self._foff,
                nbits=self._nbits,
            )
        return self._header
