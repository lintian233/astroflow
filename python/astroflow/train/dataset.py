## Label: TOA:到达时间 DM:色散量
## Input: DMT, DED_SPEC, PULSAR

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import h5py


def read_fetch_h5():
    h5_file_path = r"/data/QL/lingh/FETCH_DADASET/train_data.hdf5"
    with h5py.File(h5_file_path, "r") as f:
        # 读取数据集
        dmts = f["data_dm_time"]
        specs = f["data_freq_time"]
        lables = f["data_labels"]
        # 转换为numpy数组
        dmts = np.array(dmts, dtype=np.float32)
        specs = np.array(specs, dtype=np.float32)
        lables = np.array(lables)

    print(f"dmts shape: {dmts.shape}")
    print(f"specs shape: {specs.shape}")
    print(f"lables shape: {lables.shape}")
    return dmts, specs, lables


def save_fetch_dataset():
    dmts, specs, labels = read_fetch_h5()
    ture_label = np.where(labels == True)[0]
    output_dir = "./FETCH_DATASET/FETCH_TRAIN"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(dmts)):
        # Save the dmts and specs as images
        dmt = dmts[i]
        spec = specs[i]
        # (shape[0])

        # Normalize to 0-255 range
        dmt = cv2.normalize(dmt, None, 0, 255, cv2.NORM_MINMAX)
        spec = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to uint8 first (colormap requires uint8)
        dmt = np.uint8(dmt)
        spec = np.uint8(spec)

        # Apply colormap to grayscale images
        dmt = cv2.applyColorMap(dmt, cv2.COLORMAP_VIRIDIS)
        spec = cv2.applyColorMap(spec, cv2.COLORMAP_VIRIDIS)

        # Save to png
        cv2.imwrite(os.path.join(output_dir, f"dmt_{i}_{labels[i]}.png"), dmt)
        cv2.imwrite(os.path.join(output_dir, f"spec_{i}_{labels[i]}.png"), spec)
