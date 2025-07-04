import os


dir_path = "ql/fast_prefix_yolo_debug/candidate"

emty_dirs = []

all_dirs = os.listdir(dir_path)
for dir in all_dirs:
    dir_full_path = os.path.join(dir_path, dir)
    if not os.path.isdir(dir_full_path):
        continue
    files = os.listdir(dir_full_path)
    if len(files) == 0:
        emty_dirs.append(dir_full_path)

emty_dirs.sort()
print("Emty dirs:")
for emty_dir in emty_dirs:
    print(emty_dir)
print(f"Total emty dirs: {len(emty_dirs)}")