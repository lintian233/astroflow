## Label: TOA:到达时间 DM:色散量
## Input: DMT, DED_SPEC, PULSAR

import json
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_fetch_h5():
    h5_file_path = r"/data/QL/lingh/FETCH_DADASET/train_data.hdf5"
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


def generate_fetch_dataset_label():
    DATASET_PATH = r"/data/QL/lingh/DATASET/FETCH_TEST"
    tasks = []
    all_files = os.listdir(DATASET_PATH)
    for file in all_files:
        annotations = []
        begin = "dmt"
        end = "png"
        frbflag = "True"
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
                            "image": f"/data/local-files/?d=data/QL/lingh/DATASET/FETCH_TEST/{file}"
                        },
                        "annotations": annotations,
                    }
                )
            else:
                annotations.append({"result": []})
                tasks.append(
                    {
                        "data": {
                            "image": f"/data/local-files/?d=data/QL/lingh/DATASET/FETCH_TEST/{file}"
                        },
                        "annotations": annotations,
                    }
                )

    with open("fetch_test_label.json", "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Total images: {len(tasks)}")


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


def load_frb_simulation():
    file_path = r"/data/QL/lingh/FRB_SIMULATION_DATASET/chime_100_scaled_SNR_drifting_test_zapped_extended/DD_908_zapped_extended.npy"

    data = np.load(file_path, allow_pickle=True)
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = np.uint8(data)
    # TO RGB
    data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    print(data.shape)
    data = cv2.applyColorMap(data, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite("test.png", data)


def filter(img):
    # Apply gaussian filter 10 times
    filtered_img = img
    # Define kernel parameters
    kernel_size = 3
    kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel_2d = np.outer(kernel, kernel)

    filtered_img = cv2.filter2D(filtered_img, -1, kernel_2d)
    filtered_img = cv2.medianBlur(filtered_img, 4)

    return filtered_img


def switch_label(labels, origin_shape, new_shape):
    df = pd.DataFrame(columns=["save_name", "time_center", "dm_center", "time_left", "dm_left"])  # type: ignore
    for i, label in enumerate(labels.values):
        save_name, freq_slice, time_center, dm_center, time_left, dm_left = label
        if i < 100:
            print(save_name, freq_slice, time_center, dm_center, time_left, dm_left)

        frbflag = 1
        if time_center <= 0.0:
            frbflag = 0

        save_name = f"{save_name}_{freq_slice}_{frbflag}.png"
        # print(save_name)
        if time_center <= 0.0 or dm_center <= 0.0:
            dframe = pd.DataFrame(
                {
                    "save_name": [save_name],
                    "time_center": [-1],
                    "dm_center": [-1],
                    "time_left": [-1],
                    "dm_left": [-1],
                }
            )
            df = pd.concat([df, dframe], ignore_index=True)
            continue

        # Calculate the scaling factors
        freq_scale = new_shape[0] / origin_shape[0]
        time_scale = new_shape[1] / origin_shape[1]
        # Calculate the new values
        new_time_center = int(time_center * time_scale)
        new_dm_center = int(dm_center * freq_scale)
        new_time_width = int(time_left * time_scale)
        new_dm_width = int(dm_left * freq_scale)

        # Append the new values to the DataFrame
        dframe = pd.DataFrame(
            {
                "save_name": [save_name],
                "time_center": [new_time_center],
                "dm_center": [new_dm_center],
                "time_left": [new_time_width],
                "dm_left": [new_dm_width],
            }
        )
        df = pd.concat([df, dframe], ignore_index=True)
    print(df.head(10))
    return df


def load_draft_dataset():
    data_dir = r"/data/QL/lingh/DFRAST_DATASET/CENT_DATA/"
    out_dir = r"/data/QL/lingh/DFRAST_DATASET/CENT_DATA_DATASET/"

    all_files = os.listdir(data_dir)
    dataset_labels = load_data_label()

    os.makedirs(out_dir, exist_ok=True)
    for file in all_files:
        if not file.endswith(".npy"):
            continue
        file_path = os.path.join(data_dir, file)
        filename = file
        data = np.load(file_path, allow_pickle=True)
        labels = dataset_labels[dataset_labels["save_name"] == filename]
        # label freq_slice = i
        save_name, freq_slice, time_center, dm_center, time_left, dm_left = labels[
            labels["freq_slice"] == 0
        ].values[0]
        # print(save_name, freq_slice, time_center, dm_center, time_left, dm_left)
        frbflag = 1
        if time_center == -1.0:
            frbflag = 0

        data = data[0]
        data = filter(data)
        data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        print(data.shape)
        data = cv2.resize(data, (1024, 1024))
        data = np.uint8(data)
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        data = cv2.applyColorMap(data, cv2.COLORMAP_VIRIDIS)

        cv2.imwrite(os.path.join(out_dir, f"{filename}_{0}_{frbflag}.png"), data)


def coco_to_dabel_studio_format(coco_label, SIZE):

    WIDTH, HEIGHT = SIZE
    df = coco_label

    valid_df = df[(df["time_center"] != -1) & (df["dm_center"] != -1)]
    unvalid_df = df[(df["time_center"] == -1) | (df["dm_center"] == -1)]
    tasks = []
    for image_name, group in unvalid_df.groupby("save_name"):
        tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d=data/QL/lingh/DFRAST_DATASET/CENT_DATA_DATASET/{image_name}"
                },
                "annotations": [],
            }
        )

    for image_name, group in valid_df.groupby("save_name"):
        annotations = []
        for _, row in group.iterrows():
            x_center = row["time_center"]
            y_center = row["dm_center"]

            x_left = row["time_left"]
            y_left = row["dm_left"]
            half_width = abs(x_left - x_center)
            half_height = abs(y_left - y_center)

            x = (x_center - half_width) / WIDTH * 100
            y = (y_center - half_height) / HEIGHT * 100
            width = (2 * half_width) / WIDTH * 100
            height = (2 * half_height) / HEIGHT * 100
            # print(f"x: {x}, y: {y}, width: {width}, height: {height}")

            annotations.append(
                {
                    "result": [
                        {
                            "type": "rectanglelabels",
                            "original_width": WIDTH,
                            "original_height": HEIGHT,
                            "from_name": "label",
                            "to_name": "image",
                            "value": {
                                "x": round(x, 2),
                                "y": round(y, 2),
                                "width": round(width, 2),
                                "height": round(height, 2),
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
                    "image": f"/data/local-files/?d=data/QL/lingh/DFRAST_DATASET/CENT_DATA_DATASET/{image_name}"
                },
                "annotations": annotations,
            }
        )

    print(f"Total images: {len(tasks)}")
    with open("draft_label.json", "w") as f:
        json.dump(tasks, f, indent=2)


def load_data_label():
    file_path = r"/data/QL/lingh/DFRAST_DATASET/CENT_DATA/data_label.csv"
    df = pd.read_csv(file_path)
    # remove freq_slice != 0
    df = df[df["freq_slice"] == 0]
    print(f"Total labels: {df.shape[0]}")
    return df


def dataset_label_parser():
    # file_path = r"/data/QL/lingh/DFRAST_DATASET/CENT_DATA/data_label.csv
    df = load_data_label()
    df = switch_label(df, (1024, 8192), (1024, 1024))
    coco_to_dabel_studio_format(df, (1024, 1024))
    print(df.head(10))
    print(df.shape)


if __name__ == "__main__":
    # dataset_label_parser()
    # load_draft_dataset()
    generate_fetch_dataset_label()
