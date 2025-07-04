import multiprocessing
from astroflow import single_pulsar_search_dir
from astroflow import Config
from astroflow.config.taskconfig import TaskConfig
import argparse


# python python/main.py /data/QL/predata/7410_Crab_Base_H_25-03-24_17-06-13/ --dm_low 1 --dm_high 100 --dm_step 0.5 --freq_start 1250 --freq_end 1430 ./ql


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
