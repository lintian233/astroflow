# type: ignore
from .. import _astroflow_core as _astro_core  # type: ignore
import os

from .data import Header, SpectrumBase
from .data import SpectrumType

import your
import numpy as np


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

        self._type = SpectrumType.FIL
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
        self._continue_data = None

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
        self._data = self.core_instance.data[:, 0, :]

    @property
    def core_instance(self):
        if self._core_instance is None:
            self._load_data()
        return self._core_instance

    def get_spectrum(self):
        if self._data is None:
            self._load_data()
        return self._data
    
    def get_original_data(self) -> np.ndarray:
        if self._data is None:
            self._load_data()
        if self._continue_data is None:
            self._continue_data = self._data.reshape(-1)
        return self._continue_data

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


class FilterbankPy(SpectrumBase):

    def __init__(self, filename: str = None):
        super().__init__()

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        if os.path.splitext(filename)[1] not in [".fil", ".FIL"]:
            raise ValueError(f"Invalid file extension: {filename}")

        self._type = SpectrumType.CUSTOM
        self._filename = filename
        self._header = None
        self._continue_data = None
        self._data = None

    def _load_data(self):
        your_reader = your.Your(self._filename)
        header = your_reader.your_header
        
        fch1 = header.fch1
        foff = header.foff
        nchans = header.nchans
        
        self._data = your_reader.get_data(0, header.nspectra)

        if foff < 0:
            foff = -foff
            fch1 = fch1 - (nchans - 1) * foff
            # self._data = np.reshape(self._data, (header.npol, header.nspectra, nchans))
            # self._data = np.flip(self._data, axis=2)
            # self._data = np.ascontiguousarray(self._data, dtype=np.float32)
            self._data = np.flip(self._data, axis=1)

        self._header = Header(
            mjd=header.tstart,
            filename=self._filename,
            nifs=header.npol,
            nchans=nchans,
            ndata=header.nspectra,
            tsamp=header.tsamp,
            fch1=fch1,
            foff=foff,
            nbits=header.nbits,
        )
        
        if header.nbits == 8 and self._data.dtype != np.uint8:
            self._data = self._data.astype(np.uint8)
        elif header.nbits == 16 and self._data.dtype != np.uint16:
            self._data = self._data.astype(np.uint16)
        elif header.nbits == 32 and self._data.dtype != np.uint32:
            self._data = self._data.astype(np.uint32)

    def get_spectrum(self):
        if self._data is None:
            self._load_data()
        return self._data
    
    def get_original_data(self) -> np.ndarray:
        if self._data is None:
            self._load_data()
        if self._continue_data is None:
            self._continue_data = self._data.reshape(-1)
        return self._continue_data

    def header(self) -> Header:
        """
        Returns the header information of the filterbank file.
        """
        if self._header is None:
            self._load_data()

        return self._header
