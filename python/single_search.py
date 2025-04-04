from astroflow import single_pulsar_search_file
from astroflow import Config
import time


def main():
    config = Config(
        dm_low=0,
        dm_high=800,
        freq_start=1050,
        freq_end=1250,
        dm_step=1,
        time_downsample=1,
        t_sample=1,
    )
    file = "/home/lingh/work/astroflow/tests/FRB20241124A.fil"
    # file = "/data/QL/lingh/FAST_FRB_DATA/FRB20201124_0001.fits"
    # file = (
    #     "/data/QL/naocdata/B0534+2200/20250404/B0534+2200_20250404_170758_ant01p0.fil"
    # )
    output_dir = "frb20241124a"
    # output_dir = "B0534+2200_20250404_170758_ant01p0"
    # output_dir = "frb20201124_0001"
    single_pulsar_search_file(file, output_dir, config)


main()
