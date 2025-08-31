#! /usr/bin/env python
import argparse
import time
import os
import sys

from astroflow.logger import logger
from astroflow import Config, single_pulsar_search_file, single_pulsar_search_dir, monitor_directory_for_pulsar_search
from astroflow.config.taskconfig import TaskConfig
from astroflow.dataset.generate import count_frb_dataset

def logging_setup():
    logger.add(
        sys.stdout,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {function}- {message}",
    )

    logger.add(
        f"{TaskConfig().output}/astroflow.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {function}- {message}",
        rotation="10 MB",
        retention="10 days",
        encoding="utf-8",
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Pulsar Search with Automatic Mode Detection")
    # yaml
    parser.add_argument(
        "configfile",
        type=str,
        help="Input config file path",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=3.0,
        help="Time interval in seconds between directory checks in monitor mode (default: 3.0)"
    )
    parser.add_argument(
        "--stop-file",
        type=str,
        default=None,
        help="Path to stop file. If this file exists, monitoring will stop."
    )
    return parser.parse_args()


def astroflow_main():
    args = parse_args()
    config_file = args.configfile
    task_config = TaskConfig(config_file)
    logging_setup()

    mode = task_config.mode
    print(f"Detected mode: {mode}")
    
    if mode == "single":
        # 单文件搜索模式
        if not task_config.input or not os.path.exists(task_config.input):
            raise FileNotFoundError(f"Input file not found: {task_config.input}")
        
        if os.path.isfile(task_config.input):
            print(f"Starting single file search for: {task_config.input}")
            single_pulsar_search_file(task_config)
        else:
            raise ValueError(f"Input path is not a file: {task_config.input}")
            
    elif mode in ["directory", "muti"]:
        # 多文件/目录搜索模式
        if not task_config.input or not os.path.exists(task_config.input):
            raise FileNotFoundError(f"Input directory not found: {task_config.input}")
        
        if os.path.isdir(task_config.input):
            print(f"Starting directory search for: {task_config.input}")
            single_pulsar_search_dir(task_config)
        else:
            raise ValueError(f"Input path is not a directory: {task_config.input}")
            
    elif mode == "monitor":
        # 监控模式
        if not task_config.input or not os.path.exists(task_config.input):
            raise FileNotFoundError(f"Monitor directory not found: {task_config.input}")
        
        if os.path.isdir(task_config.input):
            print(f"Starting directory monitoring mode...")
            print(f"Monitoring directory: {task_config.input}")
            print(f"Check interval: {args.check_interval} seconds")
            if args.stop_file:
                print(f"Stop file: {args.stop_file}")
            print("Press Ctrl+C to stop monitoring")
            
            monitor_directory_for_pulsar_search(
                task_config, 
                check_interval=args.check_interval,
                stop_file=args.stop_file
            )
        else:
            raise ValueError(f"Monitor path is not a directory: {task_config.input}")
    elif mode == "dataset":
        # 数据集搜索模式
        if not task_config.candpath or not os.path.exists(task_config.candpath):
            raise FileNotFoundError(f"Candidate path not found: {task_config.candpath}")
        
        if not task_config.input or not os.path.exists(task_config.input):
            raise FileNotFoundError(f"Input directory not found: {task_config.input}")
        
        if os.path.isdir(task_config.input):
            print(f"Starting dataset search...")
            print(f"Dataset directory: {task_config.input}")
            print(f"Candidate file: {task_config.candpath}")
            
            count_frb_dataset(
                dataset_path=task_config.input,
                candidate_path=task_config.candpath,
                task_config=task_config,
            )
        else:
            raise ValueError(f"Dataset path is not a directory: {task_config.input}")
            
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are: single, directory, muti, monitor, dataset")

def main():
    try:
        astroflow_main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    exit(0)

