#!/bin/bash
set -e

# This script should be run after activating the conda environment
# Usage: conda activate dev-astroflow-ml && bash build.sh

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: Conda environment not activated. Please run:" >&2
    echo "  conda activate dev-astroflow-ml" >&2
    exit 1
fi

echo "Building astroflow..."
echo "Environment: $CONDA_PREFIX"

# Set compilers
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

echo ""
echo "Compiler configuration:"
echo "CC: $CC"
echo "CXX: $CXX"

# Verify Conan configuration
echo ""
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

# Setup build directory
build_dir="build"
if [ -d "$build_dir" ]; then
    echo ""
    echo "Cleaning old build directory..."
    rm -rf "$build_dir"
fi

if ! mkdir -p "$build_dir"; then
    echo "Error: Failed to create build directory $build_dir" >&2
    exit 1
fi

# Build
echo ""
echo "=========================================="
echo "Starting build..."
echo "=========================================="
echo ""

cd "$build_dir"

conan install .. \
    --output-folder=. \
    --build=missing \
    -s build_type=Release \
    -c tools.system.package_manager:mode=install \
    -c tools.system.package_manager:sudo=True

source conanbuild.sh

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -fno-lto" \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -fno-lto"

make -j 16

cd ..

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
