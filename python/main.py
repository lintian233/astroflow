import multiprocessing
from astroflow import single_pulsar_search_dir, monitor_directory_for_pulsar_search
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
    parser.add_argument(
        "--monitor", 
        action="store_true",
        help="Enable directory monitoring mode (check for new files every 3 seconds)"
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


def main():
    args = parse_args()
    config_file = args.configfile
    task_config = TaskConfig(config_file)
    
    if args.monitor:
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
        single_pulsar_search_dir(task_config)

main()
