import os

from .utils import Config
from .dedispered import dedispered_fil
from .frbdetector import CenterNetFrbDetector
from .plotter import PlotterManager


def single_pulsar_search(
    file: str,
    output_dir: str,
    config: Config,
    detector: CenterNetFrbDetector,
    plotter: PlotterManager,
) -> None:

    if not file.endswith(".fil"):
        raise ValueError("File must be a .fil file")
    if not os.path.exists(file):
        raise ValueError("File does not exist")

    dmtimes = dedispered_fil(
        file,
        config.dm_low,
        config.dm_high,
        config.freq_start,
        config.freq_end,
        config.dm_step,
        config.time_downsample,
        config.t_sample,
    )
    detect_dir = os.path.join(output_dir, "detect").lower()
    file_basename = os.path.basename(file).split(".")[0]
    save_path = os.path.join(output_dir, file_basename).lower()

    os.makedirs(detect_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    for idx, data in enumerate(dmtimes):
        if detector.detect(data):
            print(f"Found FRB in {file_basename} at {data}")
            plotter.plot_candidate(data, detect_dir)
            plotter.plot_dmtime(data, save_path)
        plotter.plot_dmtime(data, save_path)


def single_pulsar_search_dir(files_dir: str, output_dir: str, config: Config) -> None:

    all_files = os.listdir(files_dir)
    base_dir = os.path.basename(files_dir)

    plotter = PlotterManager()

    frb_detector = CenterNetFrbDetector(confidence=0.3)
    plotter = PlotterManager(64)
    for file in all_files:
        if not file.endswith(".fil"):
            continue

        file_dir = os.path.join(output_dir, base_dir, file.split(".")[0]).lower()

        print(f"checking {file_dir}")
        if os.path.exists(file_dir):
            print(f"跳过已处理文件: {file} (输出目录 {file_dir} 已存在)")
            continue

        file_path = os.path.join(files_dir, file)
        print(f"Processing {file_path}")
        try:
            single_pulsar_search(
                file_path,
                os.path.join(output_dir, base_dir),
                config,
                frb_detector,
                plotter,
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plotter.close()


def single_pulsar_search_file(file: str, output_dir: str, config: Config) -> None:
    plotter = PlotterManager()
    frb_detector = CenterNetFrbDetector(confidence=0.3)
    single_pulsar_search(file, output_dir, config, frb_detector, plotter)
    plotter.close()
