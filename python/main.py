import multiprocessing
from astroflow import single_pulsar_search_dir
from astroflow import Config

import argparse

# python python/main.py /data/QL/predata/7410_Crab_Base_H_25-03-24_17-06-13/ --dm_low 1 --dm_high 100 --dm_step 0.5 --freq_start 1250 --freq_end 1430 ./ql


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Run single pulsar search on a directory of filterbank files"
    )
    arg_parser.add_argument(
        "fil_dir", type=str, help="Directory containing filterbank files"
    )
    arg_parser.add_argument("output_dir", type=str, help="Output directory")
    arg_parser.add_argument(
        "--dm_low", type=float, default=5, help="Lowest DM to search"
    )
    arg_parser.add_argument(
        "--dm_high", type=float, default=1000, help="Highest DM to search"
    )
    arg_parser.add_argument("--dm_step", type=float, default=1, help="DM step size")
    arg_parser.add_argument(
        "--freq_start", type=float, default=1312, help="Start frequency in MHz"
    )
    arg_parser.add_argument(
        "--freq_end", type=float, default=1448, help="End frequency in MHz"
    )
    arg_parser.add_argument(
        "--time_downsample", type=int, default=1, help="Time downsampling factor"
    )
    arg_parser.add_argument(
        "--t_sample", type=float, default=0.5, help="Sampling time in ms"
    )

    return arg_parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        dm_low=args.dm_low,
        dm_high=args.dm_high,
        freq_start=args.freq_start,
        freq_end=args.freq_end,
        dm_step=args.dm_step,
        time_downsample=args.time_downsample,
        t_sample=args.t_sample,
    )
    single_pulsar_search_dir(args.fil_dir, args.output_dir, config)


main()
