#!/bin/bash
set -e

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

ENV_NAME="dev-astroflow-ml"

# Check if the conda environment exists
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "The '$ENV_NAME' environment exists."
else
    # Create the environment automatically if AUTO_CREATE is set or in non-interactive mode
    echo "The '$ENV_NAME' environment does not exist."
    
    # Check if we're in non-interactive mode or AUTO_CREATE is set
    if [[ "${AUTO_CREATE:-}" == "yes" ]] || [[ ! -t 0 ]]; then
        echo "Creating the '$ENV_NAME' environment automatically..."
        response="yes"
    else
        read -p "Do you want to create a new environment named '$ENV_NAME'? (yes/no): " response
    fi

    # Convert the response to lowercase for case-insensitive comparison
    response=$(echo "$response" | tr '[:upper:]' '[:lower:]')

    # Check the user's response
    if [[ "$response" == "yes" || "$response" == "y" ]]; then
        echo "Creating the '$ENV_NAME' environment with Python 3.12, numpy, and matplotlib..."
        conda create -n "$ENV_NAME" \
            --override-channels \
            python=3.12 conan numpy matplotlib \
            conda-forge::gcc==11.4.0 conda-forge::gxx==11.4.0 \
            nvidia::cuda-toolkit==12.6.0 -y
        echo "Environment created successfully."
    else
        echo "Environment creation canceled. Exiting..."
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment and build, run:"
echo "  conda activate $ENV_NAME"
echo "  bash build.sh"
echo ""
