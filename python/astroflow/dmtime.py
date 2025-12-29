class DmTime:
    def __init__(
        self,
        tstart,
        tend,
        dm_low,
        dm_high,
        dm_step,
        freq_start,
        freq_end,
        data,
        name,
    ):
        self.name = name
        self.data = data
        self.tstart = tstart
        self.tend = tend
        self.dm_low = dm_low
        self.dm_high = dm_high
        self.dm_step = dm_step
        self.freq_start = freq_start
        self.freq_end = freq_end

    def __str__(self):
        info = f"{self.name}_T_{round(self.tstart, 3)}s_{round(self.tend, 3)}s_DM_{round(self.dm_low, 3)}_{round(self.dm_high, 3)}_F_{round(self.freq_start, 3)}_{round(self.freq_end, 3)}"
        return info

    def __repr__(self):
        return self.__str__()
