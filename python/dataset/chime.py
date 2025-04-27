import json
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_frb_simulation_spec(file_path):
    data = np.load(file_path, allow_pickle=True)
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)

    data = np.uint8(data)
    data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    data = cv2.applyColorMap(data, cv2.COLORMAP_VIRIDIS)
    
    cv2.imwrite("test.png", data)


def main():
    file_path = r"/data/QL/lingh/FRB_SIMULATION_DATASET/chime_100_scaled_SNR_drifting_test_zapped_extended/DD_908_zapped_extended.npy"
    save_frb_simulation_spec(file_path)