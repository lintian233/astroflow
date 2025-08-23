import astropy.io.fits as fits
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

from .. import _astroflow_core as _astro_core  # type: ignore

uint8, uint16, uint32 = np.uint8, np.uint16, np.uint32


class Header:
    def __init__(self, mjd, filename, nifs, nchans, ndata, tsamp, fch1, foff, nbits):
        self.filename = filename
        self.mjd = mjd
        self.nifs = nifs
        self.nchans = nchans
        self.ndata = ndata
        self.tsamp = tsamp
        self.fch1 = fch1
        self.foff = foff
        self.nbits = nbits
        self._core_header = None

    def __str__(self):
        info = "--------------------------------\n"
        info += f"Filename: {self.filename}\n"
        info += f"MJD: {self.mjd}\n"
        info += f"Nr of IFs: {self.nifs}\n"
        info += f"Nr of channels: {self.nchans}\n"
        info += f"Nr of data points: {self.ndata}\n"
        info += f"Time per sample: {self.tsamp} [s]\n"
        info += f"Start frequency: {self.fch1} [MHz]\n"
        info += f"Channel bandwidth: {self.foff} [Mhz]\n"
        info += f"Nr of bits per sample: {self.nbits}\n"
        info += "--------------------------------"
        return info

    @property
    def core_header(self):
        if self._core_header is None:
            self._core_header = _astro_core.Header(self)

        return self._core_header


class SpectrumType(Enum):
    FIL = 0
    PSRFITS = 1
    CUSTOM = 2


class SpectrumBase(ABC):
    """
    Abstract base class for spectrum data handling.
    """

    def __init__(self):
        self._type = SpectrumType.CUSTOM
        self._filename = None
        self._core_data = None
        self._core_header = None

    def settype(self, spectrum_type: SpectrumType):
        """
        Set the type of the spectrum.
        """
        self._type = spectrum_type

    @abstractmethod
    def get_spectrum(self) -> np.ndarray:
        """
        Abstract method to get time-frequency spectrum data.
        """
        pass

    @abstractmethod
    def get_original_data(self) -> np.ndarray:
        """
        Abstract method to get the original data array.
        """
        pass

    @abstractmethod
    def header(self) -> Header:
        """
        Abstract method to get the header of the spectrum data.
        """
        pass

    @property
    def type(self) -> SpectrumType:
        if self._type is None:
            self._type = SpectrumType.CUSTOM
        return self._type

    @property
    def filename(self) -> str:
        if self._filename is None:
            return ""
        return self._filename

    @property
    def core_data(self) -> tuple:
        if self._core_header is not None and self._core_data is not None:
            return self._core_data, self._core_header

        header = self.header()
        self._core_data = self.get_original_data()
        self._core_header = header.core_header

        return self._core_data, self._core_header

    def spectrumset(self, spectrum: np.ndarray):
        """
        Set the spectrum data.
        """
        self._core_data = spectrum
