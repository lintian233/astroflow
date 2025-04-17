# type: ignore
import multiprocessing
import os
import time
import gc

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec

from .dedispered import dedisperse_spec_with_dm
from .dmtime import DmTime
from .io.filterbank import Filterbank
from .io.psrfits import PsrFits
from .spectrum import Spectrum

# Gaussian kernel

def error_tracer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

class PlotterManager:
    def __init__(self, max_worker=8):
        self.max_worker = max_worker
        # error log
        multiprocessing.log_to_stderr()
        self.pool = multiprocessing.Pool(self.max_worker)

    def plot_candidate(self, dmt: DmTime, candinfo, save_path, file_path):
        self.pool.apply_async(plot_candidate, args=(dmt, candinfo, save_path, file_path))

    def plot_spectrogram(self, file_path, candinfo, save_path):
        self.pool.apply_async(plot_spectrogram, args=(file_path, candinfo, save_path))

    def plot_dmtime(self, dmt: DmTime, save_path):
        self.pool.apply_async(plot_dmtime, args=(dmt, save_path))

    def plot_spec(self, spec: np.ndarray, title, info, save_path):
        self.pool.apply_async(plot_spec, args=(spec, title, info, save_path))

    def close(self):
        self.pool.close()
        self.pool.join()


def preprocess_img(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img - np.mean(img)) / np.std(img)
    img = cv2.resize(img, (512, 512))

    img = np.clip(img, *np.percentile(img, (0.1, 99.9)))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    img = seaborn.color_palette("mako", as_cmap=True)(img)
    img = img[..., :3]

    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]

    return img


def plot_spec(spec, title, candinfo, save_path, dpi=100):
    tstart = candinfo[0]
    tend = candinfo[1]
    freq_start = candinfo[2]
    freq_end = candinfo[3]
    tstart = tstart if tstart > 0 else 0
    tstart = np.round(tstart, 3)
    tend = np.round(tend, 3)

    data = spec
    fig, axs = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        dpi=dpi,
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
    )
    plt.rcParams["image.origin"] = "lower"
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    vim, vmax = np.percentile(data, [5, 95])
    time_axis = np.linspace(tstart, tend, data.shape[0])
    freq_axis = np.linspace(freq_start, freq_end, data.shape[1])
    time_series = data.sum(axis=1)
    axs[0].plot(time_axis, time_series, "k-", linewidth=0.5)
    axs[0].set_ylabel("Integrated Power")
    axs[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axs[0].set_yscale("log")
    axs[0].grid(True, alpha=0.3)
    # add title
    axs[0].set_title(f"{title}")

    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    axs[1].imshow(
        data.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=extent,
        vmin=vim,
        vmax=vmax,
    )
    axs[1].set_ylabel(f"Frequency (MHz)")
    axs[1].set_xlabel(f"Time (s)")

    axs[0].set_xlim(tstart, tend)
    axs[1].set_xlim(tstart, tend)

    plt.subplots_adjust(hspace=0.05, left=0.08, right=0.92)
    plt.savefig(
        f"{save_path}/{title}.png",
        dpi=dpi,
        bbox_inches="tight",
        format="png",
    )
    plt.close()


def plot_spectrogram(file_path, candinfo, save_path, dpi=100):
    print(f"Plotting {file_path} with {candinfo}")
    basename = os.path.basename(file_path).split(".")[0]

    origin_data = None
    if file_path.endswith(".fil"):
        origin_data = Filterbank(file_path)
    elif file_path.endswith(".fits"):
        origin_data = PsrFits(file_path)
    else:
        raise ValueError("Unknown file type")
    print(f"Loaded {file_path}")
    header = origin_data.header()

    dm = candinfo[0]
    toa = candinfo[1]
    freq_start = candinfo[2]
    freq_end = candinfo[3]

    time_size = 0.05
    tstart = toa - time_size
    tend = toa + time_size
    tstart = tstart if tstart > 0 else 0
    tstart = np.round(tstart, 3)
    tend = np.round(tend, 3)
    title = f"{basename}-spectrum-{toa}s-{tstart}s-{tend}s-{dm}pc-cm3"

    spectrum = dedisperse_spec_with_dm(
        origin_data, tstart, tend, dm, freq_start, freq_end
    )

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        dpi=dpi,
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
    )
    plt.rcParams["image.origin"] = "lower"
    data = spectrum.data

    vim, vmax = np.percentile(data, [0, 99])
    time_axis = np.linspace(tstart, tend, spectrum.ntimes)
    freq_axis = freq_start + np.arange(spectrum.nchans) * header.foff
    time_series = data.sum(axis=1)
    axs[0].plot(time_axis, time_series, "k-", linewidth=0.5)
    axs[0].set_ylabel("Integrated Power")
    axs[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axs[0].set_yscale("log")
    axs[0].grid(True, alpha=0.3)
    # add title
    axs[0].set_title(f"{title}")

    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    axs[1].imshow(
        data.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=extent,
        vmin=vim,
        vmax=vmax,
    )
    axs[1].set_ylabel(
        f"Frequency (MHz)\nFCH1={header.fch1:.3f} MHz, FOFF={header.foff:.3f} MHz"
    )
    axs[1].set_xlabel(f"Time (s)\nTSAMP={header.tsamp:.6e}s")

    axs[0].set_xlim(tstart, tend)
    axs[1].set_xlim(tstart, tend)

    plt.subplots_adjust(hspace=0.05, left=0.08, right=0.92)
    plt.savefig(
        f"{save_path}/{title}.png",
        dpi=dpi,
        bbox_inches="tight",
        format="png",
    )
    plt.close()
    print(f"Saved {save_path}/{title}.png")


def filter(img):
    kernel_size = 2
    kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel_2d = np.outer(kernel, kernel)

    for _ in range(1):
        img = cv2.filter2D(img.astype(np.float32), -1, kernel_2d)
    
    for _ in range(2):
        img = cv2.medianBlur(img.astype(np.float32), ksize=3)
    return img

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
    
    # --- 步骤1：沿频率轴积分生成时间序列 ---
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
    
    # 标记连续超过阈值的区域
    above_threshold = (time_series > threshold).astype(int)
    labeled_array, num_features = label(above_threshold)
    
    # 找到包含峰值的区域
    peak_label = labeled_array[peak_idx]
    if peak_label == 0:
        raise ValueError("峰值未超过阈值，请降低 threshold_sigma 或检查数据")
    
    signal_region = (labeled_array == peak_label)
    
    # 提取信号区域的起始和结束索引
    signal_indices = np.where(signal_region)[0]
    t_start, t_end = signal_indices[0], signal_indices[-1]
    pulse_width = t_end - t_start + 1
    
    # --- 步骤5：计算积分SNR ---
    signal_sum = np.sum(time_series[t_start:t_end+1])
    n = pulse_width  # 积分时间点数
    
    snr = (signal_sum - noise_mean * n) / (noise_std * np.sqrt(n))
    
    return snr, pulse_width, peak_idx, (noise_mean, noise_std)


def plot_dmtime(dmt: DmTime, save_path, dpi=50):
    img = dmt.data
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = filter(img)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    # cv2.guidedFilter

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img)
    img_colored = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(f"{save_path}/{dmt.__str__()}.png", img_colored)

def plot_candidate(dmt: DmTime, candinfo, save_path, file_path, dpi=150, if_clip=False, if_show=False):
    dm, toa, freq_start, freq_end = candinfo

    fig = plt.figure(figsize=(20, 10), dpi=dpi)
    # 设置height_ratios=[1, 3]，宽度不变
    gs = GridSpec(2, 4, figure=fig, width_ratios=[3, 1, 3, 1], height_ratios=[1, 3], wspace=0.04, hspace=0.04)

    # 左侧：DM-time主图和边缘图
    ax_time = fig.add_subplot(gs[0, 0])  # 上：边缘图（Integrated Power）
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_time)  # 下：主图
    ax_dm = fig.add_subplot(gs[1, 1], sharey=ax_main)    # 右：DM方向max

    dm_low = dmt.dm_low
    dm_high = dmt.dm_high
    tstart = dmt.tstart
    tend = dmt.tend
    dm_data = dmt.data
    dm_data = (dm_data - np.min(dm_data)) / (np.max(dm_data) - np.min(dm_data))
    dm_data *= 255

    dm_data = filter(dm_data)
    dm_data = cv2.resize(dm_data, (512, 512), interpolation=cv2.INTER_LINEAR)
    if if_clip:
        dm_data = np.clip(dm_data, *np.percentile(dm_data, (5, 99)))
    time_axis = np.linspace(tstart, tend, dm_data.shape[1])
    dm_axis = np.linspace(dm_low, dm_high, dm_data.shape[0])
    
    im = ax_main.imshow(
        dm_data,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[time_axis[0], time_axis[-1], dm_axis[0], dm_axis[-1]],
    )

    ax_main.set_xlabel("Time (s)", fontsize=12, labelpad=10)
    ax_main.set_ylabel("DM (pc cm$^{-3}$)", fontsize=12, labelpad=10)
    dm_sum = np.max(dm_data, axis=1)
    ax_dm.plot(dm_sum, dm_axis, lw=1.5, color="darkblue")
    ax_dm.tick_params(axis="y", labelleft=False)
    ax_dm.grid(alpha=0.3)
    time_sum = np.max(dm_data, axis=0)
    ax_time.plot(time_axis, time_sum, lw=1.5, color="darkred")
    ax_time.tick_params(axis="x", labelbottom=False)
    ax_time.grid(alpha=0.3)
    

    # 右侧：频谱主图和边缘图
    origin_data = None
    if file_path.endswith(".fil"):
        origin_data = Filterbank(file_path)
    elif file_path.endswith(".fits"):
        origin_data = PsrFits(file_path)
    else:
        raise ValueError("Unknown file type")
    header = origin_data.header()
    time_size = 0.01
    spec_tstart = toa - time_size
    spec_tend = toa + time_size
    spec_tstart = spec_tstart if spec_tstart > 0 else 0
    spec_tstart = np.round(spec_tstart, 3)
    spec_tend = np.round(spec_tend, 3)
    spectrum = dedisperse_spec_with_dm(
        origin_data, spec_tstart, spec_tend, dm, freq_start, freq_end
    )

    ax_spec_time = fig.add_subplot(gs[0, 2])  # 上：频谱边缘图
    ax_spec = fig.add_subplot(gs[1, 2], sharex=ax_spec_time)  # 下：频谱主图
    ax_spec_freq = fig.add_subplot(gs[1, 3], sharey=ax_spec)  # 右：频谱方向sum
    spec_data = spectrum.data

    snr, pulse_width, peak_idx, (noise_mean, noise_std) = calculate_frb_snr(
        spec_data, noise_range=None, threshold_sigma=5
    )
    pulse_width = pulse_width * header.tsamp * 1e3 # 转换为ms
    peak_time = spec_tstart + (peak_idx + 0.5) * header.tsamp
    print(f"TOA: {toa:.3f}s, Peak Time: {peak_time:.3f}s")
    print(f"SNR: {snr:.2f}, Pulse Width: {pulse_width:.2f} ms")

    spec_vim, spec_vmax = np.percentile(spec_data, [5, 95])
    spec_time_axis = np.linspace(spec_tstart, spec_tend, spectrum.ntimes)
    spec_freq_axis = freq_start + np.arange(spectrum.nchans) * header.foff

    spec_time_series = spec_data.sum(axis=1)
    spec_freq_series = spec_data.sum(axis=0)
    ax_spec_freq.plot(spec_freq_series, spec_freq_axis, "k-", linewidth=0.5)
    ax_spec_time.plot(spec_time_axis, spec_time_series, "k-", linewidth=0.5)
    ax_spec_time.set_ylabel("Integrated Power")
    ax_spec_time.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_spec_freq.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_spec_time.set_yscale("log")
    ax_spec_time.grid(True, alpha=0.3)

    extent_spec = [spec_time_axis[0], spec_time_axis[-1], spec_freq_axis[0], spec_freq_axis[-1]]
    spec_im = ax_spec.imshow(
        spec_data.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=extent_spec,
        vmin=spec_vim,
        vmax=spec_vmax,
    )
    ax_spec.set_ylabel(f"Frequency (MHz)\nFCH1={header.fch1:.3f} MHz, FOFF={header.foff:.3f} MHz")
    ax_spec.set_xlabel(f"Time (s)\nTSAMP={header.tsamp:.6e}s")
    ax_spec.set_xlim(spec_tstart, spec_tend)
    # 右侧色条
    # cax_spec = fig.add_axes([0.72, 0.92, 0.18, 0.02])
    # fig.colorbar(spec_im, cax=cax_spec, orientation="horizontal", label="Intensity")

    # 右上角标题
    basename = os.path.basename(file_path).split(".")[0]
    spec_title = f"{basename} - DM: {dm} pc cm$^{{-3}}$, Time: {toa:.3f}s"
    # ax_spec_time.set_title(spec_title)

    # fig.suptitle(f"{dmt.__str__()}", fontsize=16, y=0.96)
    fig.suptitle(f"FILE: {basename} - DM: {dm} - TOA: {toa:.3f}s - SNR: {snr:.2f} - Pulse Width: {pulse_width:.2f} ms - Peak Time: {peak_time:.3f}s", fontsize=16, y=0.96)
    if if_show:
        plt.show()

    output_filename = f"{save_path}/{dm}_{toa}_{dmt.__str__()}.png"
    print(f"Saving {dm}_{toa}_{dmt.__str__()}.png")
    plt.savefig(
        output_filename,
        dpi=dpi,
        format="png",
        bbox_inches="tight",
    )
    plt.close()

    # 释放文件资源
    if hasattr(origin_data, "close"):
        try:
            origin_data.close()
        except Exception:
            pass
    del origin_data
    gc.collect()