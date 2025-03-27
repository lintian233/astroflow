# type: ignore
import _astroflow_core as _astro_core
import os


class Filterbank:
    def __init__(self, file_path: str = None):
        self.core_instance = None
        if file_path is None:
            self.core_instance = _astro_core.Filterbank()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if os.path.splitext(file_path)[1] not in [".fil", ".FIL"]:
            raise ValueError(f"Invalid file extension: {file_path}")

        self.file_path = file_path
        self.core_instance = _astro_core.Filterbank(file_path)
        self.filename = self.core_instance.filename
        self.nchans = self.core_instance.nchans
        self.nifs = self.core_instance.nifs
        self.nbits = self.core_instance.nbits
        self.fch1 = self.core_instance.fch1
        self.foff = self.core_instance.foff
        self.tstart = self.core_instance.tstart
        self.tsamp = self.core_instance.tsamp
        self.ndata = self.core_instance.ndata
        self.data = self.core_instance.data


class Spectrum:
    def __init__(self, core_spectrum):
        self.data = core_spectrum.data
        self.tstart = core_spectrum.tstart
        self.tend = core_spectrum.tend
        self.dm = core_spectrum.dm
        self.ntimes = core_spectrum.ntimes
        self.nchans = core_spectrum.nchans
        self.freq_start = core_spectrum.freq_start
        self.freq_end = core_spectrum.freq_end
