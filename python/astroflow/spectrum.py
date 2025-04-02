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
