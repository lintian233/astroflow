from astroflow import single_pulsar_search_file
from astroflow import Config
import time


def main():
    config = Config(
        dm_low=0,
        dm_high=800,
        freq_start=1000,
        freq_end=1250,
        dm_step=1,
        time_downsample=1,
        t_sample=0.3,
    )
    # file = "/home/lingh/work/astroflow/tests/FRB20241124A.fil"
    file = "/data/QL/lingh/FAST_FRB_DATA/FRB20201124_0001.fits"
    # output_dir = "frb20241124a"
    output_dir = "frb20201124_0001"
    single_pulsar_search_file(file, output_dir, config)
    time.sleep(5)


main()
