import argparse
import time

from astroflow import Config, single_pulsar_search_file
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
    config_file = args.configfile
    task_config = TaskConfig(config_file)
    single_pulsar_search_file(task_config)


main()
