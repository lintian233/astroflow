#!/bin/bash
echo "this script will configure the all env for the astroflow dev"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found in PATH. Aborting!" >&2
    exit 127
fi

CONDA_PREFIX=$(conda info --base)
if [ ! -w "$CONDA_PREFIX" ]; then
    echo "Error: No write permission for conda installation at $CONDA_PREFIX" >&2
    exit 1
fi

echo "Detected conda installation: $(which conda)"

ENV_NAME="dev-astroflow"

# Check if the conda environment exists
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "The '$ENV_NAME' environment exists. Activating it..."
    conda activate "$ENV_NAME"
else
    # Prompt the user to create the environment
    echo "The '$ENV_NAME' environment does not exist."
    read -p "Do you want to create a new environment named '$ENV_NAME'? (yes/no): " response

    # Convert the response to lowercase for case-insensitive comparison
    response=$(echo "$response" | tr '[:upper:]' '[:lower:]')

    # Check the user's response
    if [[ "$response" == "yes" || "$response" == "y" ]]; then
        echo "Creating the '$ENV_NAME' environment with Python 3.12, numpy, and matplotlib..."
        conda create -n "$ENV_NAME" \
            --override-channels \
            --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
            --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
            python=3.13.2 conan numpy matplotlib gcc gxx cuda-toolkit -y
        echo "Environment created. Activating it..."
        conda activate "$ENV_NAME"
    else
        echo "Environment creation canceled. Exiting..."
        exit 1
    fi
fi

PYTHON_BIN_DIR=$CONDA_PREFIX/bin
echo "PYTHON_BIN_DIR: $PYTHON_BIN_DIR"
export PATH="$PYTHON_BIN_DIR:$PATH"
export CC=$(which gcc)
export CXX=$(which g++)

echo "Verifying Conan configuration:"
conan --version

if ! conan profile list | grep -q "default"; then
    echo "Initializing Conan profile for first-time use..."
    if ! conan profile detect --force; then
        echo "Error: Failed to initialize Conan profile" >&2
        exit 1
    fi
fi

echo "Current Conan profiles:"
conan profile show

build_dir="build"
if [ -d "$build_dir" ]; then
    if [ ! -w "$build_dir" ]; then
        echo "Error: Build directory $build_dir exists but is not writable" >&2
        exit 1
    fi
else
    if ! mkdir -p "$build_dir"; then
        echo "Error: Failed to create build directory $build_dir" >&2
        exit 1
    fi
fi

cd "$build_dir" && \
conan install .. --output-folder=. --build=missing && \
source conanbuild.sh && \
cmake .. -DCMAKE_BUILD_TYPE=Release && \
make -j 8 && \
cd ..
