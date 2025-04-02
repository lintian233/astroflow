import os
import time
import tqdm

from .utils import Config
from .dedispered import dedispered_fil
from .frbdetector import CenterNetFrbDetector
from .plotter import PlotterManager
from .io.filterbank import Filterbank
from .io.psrfits import PsrFits
from .dedispered import dedisperse_spec


def single_pulsar_search(
    file: str,
    output_dir: str,
    config: Config,
    detector: CenterNetFrbDetector,
    plotter: PlotterManager,
) -> None:

    origin_data = None
    if file.endswith(".fil"):
        origin_data = Filterbank(file)
    elif file.endswith(".fits"):
        origin_data = PsrFits(file)
    else:
        raise ValueError("Unknown file type")

    dmtimes = dedisperse_spec(
        origin_data,
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
        candidate = detector.detect(data)
        for candinfo in candidate:
            print(f"Found FRB in {file_basename} at {candinfo}")
            plotter.plot_candidate(data, detect_dir)
            plotter.plot_spectrogram(file, candinfo, detect_dir)


def single_pulsar_search_dir(files_dir: str, output_dir: str, config: Config) -> None:

    if files_dir[-1] == "/":
        files_dir = files_dir[:-1]
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]

    all_files = os.listdir(files_dir)
    base_dir = os.path.basename(files_dir)
    base_dir += f"-{config.dm_low}DM-{config.dm_high}DM"
    base_dir += f"-{config.freq_start}MHz-{config.freq_end}MHz"

    plotter = PlotterManager()

    frb_detector = CenterNetFrbDetector(confidence=0.4)
    plotter = PlotterManager(6)
    for file in tqdm.tqdm(all_files):
        if not file.endswith(".fil") and not file.endswith(".fits"):
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
