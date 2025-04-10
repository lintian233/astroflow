from astroflow.search import single_pulsar_search_with_dm_file
from astroflow.utils import SingleDmConfig

import argparse


def parse_args():
    argparse.ArgumentParser(description="Single Pulsar Search")
    parser = argparse.ArgumentParser(description="Single Pulsar Search")
    parser.add_argument(
        "file",
        type=str,
        help="Input file path",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for the results",
    )
    parser.add_argument(
        "--dm",
        type=float,
        default=40,
        help="DM value for search",
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
        "--t_sample",
        type=float,
        default=0.1,
        help="Sample time for search",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = SingleDmConfig(
        dm=args.dm,
        freq_start=args.freq_start,
        freq_end=args.freq_end,
        t_sample=args.t_sample,
    )
    single_pulsar_search_with_dm_file(
        args.file,
        args.output_dir,
        config,
    )


if __name__ == "__main__":
    main()
