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
        info = f"{self.name}_T_{self.tstart}s_{self.tend}s_DM_{self.dm_low}_{self.dm_high}_F_{self.freq_start}_{self.freq_end}"
        return info

    def __repr__(self):
        return self.__str__()
