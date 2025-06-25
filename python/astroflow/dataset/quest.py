import os
import json
import re
import numpy as np


def parser_candidate_png_path(dir_path):
    """
    Parse the candidate png path from the given directory path.
    Args:
        dir_path (str): The directory path to parse.
    Returns:
        list: A list of candidate png paths.
    """
    files = os.listdir(dir_path)

    png_files = [f for f in files if f.endswith(".png")]

    png_files.sort()

    png_paths = [os.path.join(dir_path, f) for f in png_files]
    return png_paths


def generate_label_studio_labels_json(png_paths, save_name):
    """
    Generate a JSON file for Label Studio labels from the given PNG paths.
    Args:
        png_paths (list): A list of PNG paths.
        save_name (str): The name of the JSON file to save.
    """
    labels = []
    for png_path in png_paths:
        annotation = []
        # /path/to/image.png -> path/to/image.png
        png_path = png_path[1:]
        png_path = f"/data/local-files/?d={png_path}"
        annotation.append({"result": []})
        labels.append(
            {
                "data": {
                    "image": png_path,
                },
                "annotations": annotation,
            }
        )

    # Save the labels to a JSON file
    with open(f"{save_name}.json", "w") as f:
        json.dump(labels, f, indent=2)


def main():
    # Example usage
    dir_path = "/data/QL/lingh/ql-dmt/naoc_20250416_165723_gm12"
    save_name = "quest_naoc_20250416_165723"
    png_paths = parser_candidate_png_path(dir_path)
    generate_label_studio_labels_json(png_paths, save_name)


if __name__ == "__main__":
    main()
