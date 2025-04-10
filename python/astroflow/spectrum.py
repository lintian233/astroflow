class Spectrum:
    def __init__(self, data, tstart, tend, dm, ntimes, nchans, freq_start, freq_end):
        self.data = data
        self.tstart = tstart
        self.tend = tend
        self.dm = dm
        self.ntimes = ntimes
        self.nchans = nchans
        self.freq_start = freq_start
        self.freq_end = freq_end

    @staticmethod
    def from_core_spectrum(core_spectrum):
        return Spectrum(
            data=core_spectrum.data,
            tstart=core_spectrum.tstart,
            tend=core_spectrum.tend,
            dm=core_spectrum.dm,
            ntimes=core_spectrum.ntimes,
            nchans=core_spectrum.nchans,
            freq_start=core_spectrum.freq_start,
            freq_end=core_spectrum.freq_end,
        )

    def clip(self, t_sample):
        t_len = self.tend - self.tstart
        n_samples = int(t_len / t_sample)
        remaining_samples = self.ntimes % n_samples
        data = self.data
        if remaining_samples > 0:
            data = data[:-remaining_samples]
        folded_data = data.reshape(n_samples, -1, self.nchans)
        return folded_data
