import multiprocessing
from astroflow import single_pulsar_search_dir
from astroflow import Config

import argparse


def main():
    config = Config(
        dm_low=0,
        dm_high=300,
        freq_start=1312,
        freq_end=1448,
        dm_step=1,
        time_downsample=1,
        t_sample=0.5,
    )
    fil_dir = "/data/QL/predata/7400_FRB20250316Spec_25-03-20_21-40-40"
    output_dir = "."
    single_pulsar_search_dir(fil_dir, output_dir, config)


main()
