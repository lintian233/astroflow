import your
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a spectrum from a filterbank file."
    )
    parser.add_argument("filename", type=str, help="Path to the filterbank file.")
    parser.add_argument(
        "--start", type=float, default=0.0, help="Start time in seconds."
    )
    parser.add_argument("--end", type=float, default=1.0, help="End time in seconds.")
    parser.add_argument(
        "--freq_start", type=float, default=0.0, help="Start frequency in MHz."
    )
    parser.add_argument(
        "--freq_end", type=float, default=1.0, help="End frequency in MHz."
    )
    return parser.parse_args()


def plot_spectrum(
    filename, start_time, end_time, freq_start, freq_end, dpi=300, mask=None
):

    your_reader = your.Your(filename)
    header = your_reader.your_header
    print("header", header)
    tstart = int(start_time / header.tsamp)
    tend = int(end_time / header.tsamp)
    ndata = tend - tstart
    print("tstart", tstart)
    print("tend", tend)
    data = your_reader.get_data(tstart, ndata)
    foff = header.foff
    fch1 = header.fch1
    if foff < 0:
        fch1 = header.fch1 + (header.nchans - 1) * foff
        foff = -foff
        data = data[:, ::-1]

    chan_start = int((freq_start - fch1) / foff)
    chan_end = int((freq_end - fch1) / foff)
    data = data[:, chan_start:chan_end]

    # Apply frequency mask if provided
    if mask is not None:
        # Validate mask dimensions
        mask = mask[chan_start:chan_end]
        # 转化成2D chans, times
        mask = np.repeat(mask[:, np.newaxis], ndata, axis=1)
        mask = mask.reshape((mask.shape[0], mask.shape[1]))
        mask = mask.T
        if mask.ndim != 2 or mask.shape != data.shape:
            print(
                f"Invalid mask dimensions {mask.shape}, expected {(data.T.shape)}. Skipping mask."
            )
        else:
            # Calculate 25th percentile of masked data
            # 生成高斯噪声代替原中值填充
            noise_mean = np.median(data)
            noise_std = 5  # 标准差
            noise_shape = data.T.shape

            # 创建噪声矩阵并应用mask区域
            gaussian_noise = np.random.normal(noise_mean, noise_std, noise_shape)
            data = np.where(mask.T, gaussian_noise, data.T).T
            print(f"Applied frequency mask to {np.sum(mask)} channels")

    ndata, nchans = data.shape
    print("ndata", ndata)
    print("nchans", nchans)
    plt.figure(figsize=(10, 5), dpi=dpi)
    freqs = np.arange(chan_start, chan_end) * foff + fch1
    times = np.arange(ndata) * header.tsamp + start_time
    # 指数变换增加对比度
    # 归一化
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    vmin, vmax = np.percentile(data, [1, 99.9])
    plt.imshow(
        data.T,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")

    savename = filename.split("/")[-1].split(".")[0]
    if mask is not None:
        savename += "_masked"

    plt.savefig(
        f"{savename}_{start_time}_{end_time}_{freq_start}_{freq_end}.png", dpi=dpi
    )
    plt.close()
    print(
        f"Saved plot to {savename}_{start_time}_{end_time}_{freq_start}_{freq_end}.png"
    )


def main():
    args = parse_args()

    # 创建频率范围mask
    your_reader = your.Your(args.filename)
    header = your_reader.your_header
    foff = header.foff
    fch1 = header.fch1
    if foff < 0:
        fch1 = header.fch1 + (header.nchans - 1) * foff
        foff = -foff

    # 计算屏蔽通道范围（示例使用1250-1280MHz）
    mask_chan_start = int((1266 - fch1) / foff)
    mask_chan_end = int((1272 - fch1) / foff)

    # 创建二维mask（所有时间点都屏蔽这些通道）
    mask = np.zeros((header.nchans, 1), dtype=bool)  # 初始化为False
    mask[mask_chan_start : mask_chan_end + 1] = True  # 设置目标通道为True

    plot_spectrum(
        args.filename, args.start, args.end, args.freq_start, args.freq_end, mask=mask
    )


if __name__ == "__main__":
    main()
