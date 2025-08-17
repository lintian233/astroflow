import multiprocessing
from astroflow import single_pulsar_search_dir
from astroflow import Config
from astroflow.config.taskconfig import TaskConfig
import argparse


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
    args = parse_args()
    config_file = args.configfile
    task_config = TaskConfig(config_file)
    single_pulsar_search_dir(task_config)


main()
