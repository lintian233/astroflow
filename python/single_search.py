from astroflow import single_pulsar_search_file
from astroflow import Config
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Single Pulsar Search")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Input file path",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for the results",
    )
    parser.add_argument(
        "--dm_low",
        type=float,
        default=40,
        help="Lower DM value for search",
    )
    parser.add_argument(
        "--dm_high",
        type=float,
        default=70,
        help="Upper DM value for search",
    )
    parser.add_argument(
        "--freq_start",
        type=float,
        default=1250,
        help="Start frequency for search",
    )
    parser.add_argument(
        "--freq_end",
        type=float,
        default=1430,
        help="End frequency for search",
    )
    parser.add_argument(
        "--dm_step",
        type=float,
        default=0.2,
        help="DM step size for search",
    )
    parser.add_argument(
        "--time_downsample",
        type=int,
        default=1,
        help="Time downsample factor",
    )

    parser.add_argument(
        "--t_sample",
        type=float,
        default=0.05,
        help="Time sample size",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence level for detection",
    )

    return parser.parse_args()


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
        confidence=args.confidence,
    )
    file = args.file
    output_dir = args.output_dir
    single_pulsar_search_file(file, output_dir, config)


main()
