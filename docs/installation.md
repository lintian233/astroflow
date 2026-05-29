# Installation

## Requirements

For the Python package:

- Python 3.10 to 3.12 is recommended.
- Linux is the primary supported platform.
- NVIDIA GPU and a compatible CUDA runtime are required for GPU-accelerated search.

For source builds:

- CUDA Toolkit 12.6 or newer.
- GCC/G++ 11.4 or compatible compiler.
- CMake 3.22 or newer.
- Conan 2.x.
- Ninja.

## Install From PyPI

```bash
pip install pulseflow
```

The installed command-line entry point is:

```bash
astroflow path/to/config.yaml
```

Additional command-line tools:

```bash
astrorfimask --help
specplot --help
```

## Install a Pre-Release Build

TestPyPI receives newer development builds more frequently:

```bash
pip install --pre -i https://test.pypi.org/simple/ pulseflow --extra-index-url https://pypi.org/simple
```

## Build From Source

```bash
git clone https://github.com/lintian233/astroflow
cd astroflow
bash configure.sh
conda activate dev-astroflow-ml
bash build.sh
pip install -e .
```

## First-Run Model Download

When `modelpath` is omitted, AstroFlow downloads the default YOLO model to:

```text
~/.config/astroflow/yolo11n_0816_v1.pt
```

If the automatic download fails, download the model manually:

```bash
mkdir -p ~/.config/astroflow
wget -O ~/.config/astroflow/yolo11n_0816_v1.pt \
  "https://github.com/lintian233/astroflow/releases/download/v0.1.1/yolo11n_0816_v1.pt"
```
