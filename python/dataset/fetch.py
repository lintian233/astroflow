import json
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_fetch_dataset(h5_file_path):
    with h5py.File(h5_file_path, "r") as f:
        # 读取数据集
        dmts = f["data_dm_time"]
        specs = f["data_freq_time"]
        lables = f["data_labels"]
        dmts = np.array(dmts, dtype=np.float32)
        specs = np.array(specs, dtype=np.float32)
        lables = np.array(lables)

    print(f"dmts shape: {dmts.shape}")
    print(f"specs shape: {specs.shape}")
    print(f"lables shape: {lables.shape}")
    return dmts, specs, lables


def generate_fetch_dataset_label(dataset_path, save_name):
    
    tasks = []
    all_files = os.listdir(dataset_path)
    for file in all_files:
        annotations = []
        begin = "dmt"
        end = "png"
        frbflag = "True"
        
        dir_path = dataset_path[1:]
        images_path = f"/data/local-files/?d={dir_path}/{file}"

        if begin in file and end in file:
            if frbflag in file:
                annotations.append(
                    {
                        "result": [
                            {
                                "type": "rectanglelabels",
                                "original_width": 256,
                                "original_height": 256,
                                "from_name": "label",
                                "to_name": "image",
                                "value": {
                                    "x": 40,
                                    "y": 40,
                                    "width": 20,
                                    "height": 20,
                                    "rotation": 0,
                                    "rectanglelabels": ["object"],
                                },
                            }
                        ]
                    }
                )
                tasks.append(
                    {
                        "data": {
                            "image": images_path, 
                        },
                        "annotations": annotations,
                    }
                )
            else:
                annotations.append({"result": []})
                tasks.append(
                    {
                        "data": {
                            "image": images_path,
                        },
                        "annotations": annotations,
                    }
                )

    with open(f"{save_name}.json", "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Total images: {len(tasks)}")

def save_fetch_dataset(dmts, specs, labels, save_dir):
    ture_label = np.where(labels == True)[0]
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(dmts)):
        # Save the dmts and specs as images
        dmt = dmts[i]
        spec = specs[i]

        # Normalize to 0-255 range
        dmt = np.clip(dmt, *np.percentile(dmt, (1, 99.5)))
        dmt = cv2.normalize(dmt, None, 0, 255, cv2.NORM_MINMAX)
        spec = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to uint8 first (colormap requires uint8)
        dmt = np.uint8(dmt)
        spec = np.uint8(spec)

        # Apply colormap to grayscale images
        dmt = cv2.applyColorMap(dmt, cv2.COLORMAP_VIRIDIS)
        spec = cv2.applyColorMap(spec, cv2.COLORMAP_VIRIDIS)

        # Save to png
        cv2.imwrite(os.path.join(save_dir, f"dmt_{i}_{labels[i]}.png"), dmt)
        cv2.imwrite(os.path.join(save_dir, f"spec_{i}_{labels[i]}.png"), spec)


def main():
    h5_file_path = r"/data/QL/lingh/FETCH_DADASET/test_data.hdf5"
    dmts, specs, lables = load_fetch_dataset(h5_file_path)

    dataset_path = r"/data/QL/lingh/DATASET/FETCH_CLIP_TEST"
    save_fetch_dataset(dmts, specs, lables, dataset_path)

    # generate_fetch_dataset_label(dataset_path, "fetch_test_label")

if __name__ == "__main__":
    main()

