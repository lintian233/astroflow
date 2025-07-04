from astroflow import dedispered_fil_with_dm, Spectrum, Filterbank
from astroflow.dedispered import dedisperse_spec_with_dm
from astroflow.io.filterbank import Filterbank
from astroflow.io.psrfits import PsrFits

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot dedispersed spectrum")
    parser.add_argument("file_path", type=str, help="Path to the filterbank file")
    parser.add_argument("toa", type=float, help="Start time in seconds")
    parser.add_argument("tband", type=float, default=0.1, help="Time band in seconds")
    parser.add_argument("dm", type=float, help="Dispersion measure in pc cm^-3")
    parser.add_argument("output_path", type=str, help="Path to save the plot")
    parser.add_argument(
        "--freq_start", type=float, default=-1, help="Start frequency in MHz"
    )
    parser.add_argument(
        "--freq_end", type=float, default=-1, help="End frequency in MHz"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to the mask file (optional)",
    )
    return parser.parse_args()


import numpy as np
from scipy.ndimage import label


def calculate_frb_snr(spec, noise_range=None, threshold_sigma=5.0):
    """
    计算FRB信号在二维频谱图上的信噪比（SNR）

    参数：
    spec : 2D numpy数组
        输入的二维频谱图，形状为 (时间点, 频率通道)
    noise_range : list of tuples 或 None, 可选
        噪声区域的时间范围，例如 [(0, 100), (900, 1000)] 表示取前100和最后100个时间点作为噪声
        如果为None，自动选择远离峰值的前后20%区域
    threshold_sigma : float, 可选
        判断信号区域的阈值（单位：噪声标准差）

    返回：
    snr : float
        积分信噪比（SNR）
    pulse_width : int
        脉冲宽度（时间点数）
    peak_idx : int
        峰值位置的时间索引
    noise_stats : tuple
        噪声的均值(mean)和标准差(std)
    """

    time_series = np.sum(spec, axis=1)  # 假设时间轴为第0维

    # --- 步骤2：自动选择噪声区域 ---
    n_time = len(time_series)
    if noise_range is None:
        # 默认取前后20%的时间点作为噪声区域（避开中间可能的信号）
        margin = int(0.2 * n_time)
        noise_slices = [slice(0, margin), slice(-margin, None)]
    else:
        # 根据用户指定的范围生成切片
        noise_slices = [slice(start, end) for (start, end) in noise_range]

    # 合并所有噪声区域的数据
    noise_data = np.concatenate([time_series[s] for s in noise_slices])
    noise_mean = np.mean(noise_data)
    noise_std = np.std(noise_data)

    # --- 步骤3：检测峰值位置 ---
    peak_idx = np.argmax(time_series)  # 峰值时间索引

    # --- 步骤4：确定脉冲宽度 ---
    # 计算动态阈值（基于噪声统计）
    threshold = noise_mean + threshold_sigma * noise_std
    print(f"Threshold: {threshold:.2f} (mean={noise_mean:.2f}, std={noise_std:.2f})")
    # 标记连续超过阈值的区域
    above_threshold = (time_series > threshold).astype(int)
    labeled_array, num_features = label(above_threshold)

    # 找到包含峰值的区域
    peak_label = labeled_array[peak_idx]
    if peak_label == 0:
        print("Warning: Peak not found in labeled array, check threshold or data")
        raise ValueError("峰值未超过阈值，请降低 threshold_sigma 或检查数据")

    signal_region = labeled_array == peak_label

    # 提取信号区域的起始和结束索引
    signal_indices = np.where(signal_region)[0]
    t_start, t_end = signal_indices[0], signal_indices[-1]
    pulse_width = t_end - t_start + 1

    # --- 步骤5：计算积分SNR ---
    signal_sum = np.sum(time_series[t_start : t_end + 1])
    n = pulse_width  # 积分时间点数

    snr = (signal_sum - noise_mean * n) / (noise_std * np.sqrt(n))

    return snr, pulse_width, peak_idx, (noise_mean, noise_std)


def plot_ded_spectrum(
    file_path, toa, tband, dm, output_path, freq_start=-1, freq_end=-1, mask=None
):
    tstart = toa - tband / 2
    tend = toa + tband / 2
    tstart = max(0, tstart)
    tend = min(tend, 1e10)  # Set a reasonable upper limit for tend
    tstart = round(tstart, 5)
    tend = round(tend, 5)
    os.makedirs(output_path, exist_ok=True)
    basename = os.path.basename(file_path)
    title = f"{basename}-{tstart}s-{tend}s-{dm}pc-cm3"
    origin_data = None

    if file_path.endswith(".fil"):
        origin_data = Filterbank(file_path)
    elif file_path.endswith(".fits"):
        origin_data = PsrFits(file_path)
    else:
        raise ValueError("Unknown file type")

    spectrum = dedisperse_spec_with_dm(
        origin_data, tstart, tend, dm, freq_start, freq_end
    )
    header = origin_data.header()

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        dpi=100,
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
    )
    plt.rcParams["image.origin"] = "lower"

    data = spectrum.data
    # snr, pulse_width, peak_idx, (noise_mean, noise_std) = calculate_frb_snr(
    #     data, noise_range=None, threshold_sigma=5
    # )
    # print(f"SNR: {snr:.2f}")
    # print(f"Pulse Width: {pulse_width} time points")
    if mask is not None:
        print(data.shape)
        data[:, mask[:-1]] = 0

    # vmin, vmax = np.percentile(spectrum.data, [5, 95])
    vmin, vmax = np.percentile(data, [5, 99])
    if vmin == 0:
        non_zero_values = data[data > 0]
        if non_zero_values.size > 0:
            vmin = non_zero_values.min()
        
    print(f"vmin: {vmin}, vmax: {vmax}")
    time_axis = np.linspace(tstart, tend, spectrum.ntimes)
    freq_axis = freq_start + np.arange(spectrum.nchans) * header.foff
    time_series = data.sum(axis=1)  # 通道积分
    # 高斯卷积time_series
    # time_series = cv2.GaussianBlur(time_series, (3, 1), 0)
    axs[0].plot(time_axis, time_series, "k-", linewidth=0.5)
    axs[0].set_ylabel("Integrated Power")
    axs[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axs[0].set_yscale("log")
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title(f"{title}")

    # 设置imshow的显示范围和方向
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    axs[1].imshow(
        data.T,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    axs[1].set_ylabel(
        f"Frequency (MHz)\nFCH1={header.fch1:.3f} MHz, FOFF={header.foff:.3f} MHz"
    )
    axs[1].set_xlabel(f"Time (s)\nTSAMP={header.tsamp:.6e}s")

    axs[0].set_xlim(tstart, tend)
    axs[1].set_xlim(tstart, tend)
    plt.subplots_adjust(hspace=0.05, left=0.08, right=0.92)

    source_nane = file_path.split("/")[-1].split(".")[0]
    plt.savefig(
        f"{output_path}/{title}.png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
        format="png",
    )
    print(f"Saved {output_path}/{title}.png")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    print(f"args: {args}")
    mask_file = args.mask
    mask = None
    if mask_file and os.path.exists(mask_file):
        with open(mask_file, "r") as f:
            data = f.read()

        bad_channels = list(map(int, data.split()))

        mask = np.array(bad_channels)

    plot_ded_spectrum(
        args.file_path,
        args.toa,
        args.tband,
        args.dm,
        args.output_path,
        args.freq_start,
        args.freq_end,
        mask=mask,
    )
