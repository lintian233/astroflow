import argparse

from astroflow.search import single_pulsar_search_with_dm_dir
from astroflow.utils import SingleDmConfig, Config
from astroflow.config.taskconfig import TaskConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Single Pulsar Search")
    # yaml
    parser.add_argument(
        "configfile",
        type=str,
        help="Input config file path",
    )
    return parser.parse_args()

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
