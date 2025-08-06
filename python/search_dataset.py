import os

from astroflow.dataset.generate import generate_frb_dataset, count_frb_dataset
from astroflow.config.taskconfig import TaskConfig

import argparse

def parse_args():
    # taskconfig
    parser = argparse.ArgumentParser(description="Generate FRB dataset")
    parser.add_argument(
        "taskconfig",
        type=str,
        help="Path to the task configuration file",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    taskconfig_filepath = parse_args().taskconfig
    taskconfig = TaskConfig(taskconfig_filepath)
    count_frb_dataset(
        dataset_path=taskconfig.input,
        candidate_path="/home/lingh/work/astroflow/candidate_check.csv",
        task_config=taskconfig,
    )