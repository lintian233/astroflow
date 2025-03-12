# type: ignore

import astroflow

import numpy as np
import matplotlib.pyplot as plt

# # 生成100万个0-100的随机浮点数
# a = np.random.uniform(0, 100, size=1_000_000).astype(np.float32)
# b = np.random.uniform(0, 100, size=1_000_000).astype(np.float32)
#
# result = demo.VectorAdder.add_vectors(a, b)
#
# arr = demo.get_data()
# print(arr.__array_interface__["data"][0])  # 查看内存地址
# assert arr.nbytes == 1000000 * 2  # 验证内存大小
# print(arr.nbytes)
#
# # 测试Python到C++的零拷贝
# original = np.arange(100000, dtype=np.uint16)
# demo.process_data(original)
# for i in range(10):
#     print(original[i])

file_path = r"/home/lingh/work/astroflow/tests/FRB20241124A.fil"

time_downsample = 1
dm_low = 0
dm_high = 800
freq_start = 1130
freq_end = 1300
dm_step = 1
t_sample = 0.5

data = astroflow.dedisper_fil_uint8(
    file_path,
    dm_low,
    dm_high,
    freq_start,
    freq_end,
    dm_step,
    time_downsample,
    t_sample,
    njobs=120,
)


def plot_dedispered_data(dm_data, title):
    time_axis = np.linspace(0, data.tsample * dm_data.shape[1], dm_data.shape[1])
    dm_axis = np.linspace(data.dm_low, data.dm_high, dm_data.shape[0])

    plt.figure(figsize=(12, 8), dpi=300)
    X, Y = np.meshgrid(time_axis, dm_axis)

    im = plt.pcolormesh(X, Y, dm_data, shading="auto", cmap="viridis")

    plt.xlabel("Time (tsample)", fontsize=12, labelpad=10)
    plt.ylabel("DM (pc cm$^{-3}$)", fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, pad=15)

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Signal Strength (arb. units)", rotation=270, labelpad=15)

    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


for idx, dm in enumerate(data.dm_times):
    plot_dedispered_data(
        dm.reshape(data.shape[0], data.shape[1]),
        f"Dedispersed-Profile-DM-Trial-{idx+1}",
    )
