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
        self.tstart = round(tstart, 3)
        self.tend = round(tend, 3)
        self.dm_low = round(dm_low, 3)
        self.dm_high = round(dm_high, 3)
        self.dm_step = round(dm_step, 3)
        self.freq_start = round(freq_start, 3)
        self.freq_end = round(freq_end, 3)

    def __str__(self):
        info = f"{self.name}_T_{self.tstart}s_{self.tend}s_DM_{self.dm_low}_{self.dm_high}_F_{self.freq_start}_{self.freq_end}"
        return info

    def __repr__(self):
        return self.__str__()
