import os

from astroflow.dataset.generate import generate_frb_dataset


def main():
    png_path = "/data/QL/lingh/FAST_PREFIX/PNG"
    label_path = "/data/QL/lingh/FAST_PREFIX/LABEL/FRB20201124.json"
    dataset_path = "/data/QL/lingh/FAST_FRB_DATA"
    candidate_path = "/data/QL/lingh/FAST_FRB_DATA/FRB20201124_summary.csv"
    os.makedirs(png_path, exist_ok=True)
    label_dir = os.path.dirname(label_path)
    print(label_dir)
    os.makedirs(label_dir, exist_ok=True)
    dm_low = 100
    dm_high = 800
    dm_step = 1
    freq_start = 1000
    freq_end = 1250
    t_sample = 0.5 #s
    time_downsample = 1
    generate_frb_dataset(
        dataset_path,
        candidate_path,
        dm_low,
        dm_high,
        dm_step,
        freq_start,
        freq_end,
        t_sample,
        label_path,
        png_path,
        time_downsample
    )

if __name__ == "__main__":
    main()
