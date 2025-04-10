import argparse

from astroflow.search import single_pulsar_search_with_dm_dir
from astroflow.utils import SingleDmConfig, Config


def parse_args():
    argparser = argparse.ArgumentParser(description="Single Pulsar Search with DM DIR")
    argparser.add_argument(
        "files_dir",
        type=str,
        help="Directory containing input files",
    )
    argparser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save output files",
    )
    argparser.add_argument(
        "--dm",
        type=float,
        default=40,
        help="DM value for search",
    )
    argparser.add_argument(
        "--t_sample",
        type=float,
        default=0.1,
        help="Time sample for search",
    )
    argparser.add_argument(
        "--freq_start",
        type=float,
        default=1250,
        help="Start frequency for search",
    )
    argparser.add_argument(
        "--freq_end",
        type=float,
        default=1430,
        help="End frequency for search",
    )
    return argparser.parse_args()


def main():
    args = parse_args()
    files_dir = args.files_dir
    output_dir = args.output_dir
    config = SingleDmConfig(
        dm=args.dm,
        freq_start=args.freq_start,
        freq_end=args.freq_end,
        t_sample=args.t_sample,
    )
    single_pulsar_search_with_dm_dir(files_dir, output_dir, config)


if __name__ == "__main__":
    main()
