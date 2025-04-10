# AstroFlow 🌌

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/lintian233/astroflow/build.yml?logo=github)](https://github.com/lintian233/astroflow/actions)
[![C++ Standard](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)](https://isocpp.org/)
[![Python Version](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://www.python.org/)
[![Conda Supported](https://img.shields.io/conda/v/conda-forge/python?logo=anaconda)](https://conda.io/)

A high-performance radio astronomy data processing framework combining C++ computational efficiency with Python usability, featuring GPU acceleration and automated build configuration.

## ✨ Key Features
- 🚀 CUDA-accelerated numerical core
- 📊 Interactive visualization with Matplotlib
- 🔧 Automated environment configuration
- 📦 Cross-platform dependency management (Conan + Conda)
- 🧪 Comprehensive testing framework

## 🚀 Quick Start
```bash
# Clone repository and create feature branch
git clone https://github.com/lintian233/astroflow.git
cd astroflow

# Run automated configuration
source configure.sh
pip install -r python/requirements.txt

# Test if sucessful installed
cd build && ctest 
cd python && python -m unittest
```
⚠️ **Note** The configuration script will:
1. Create a Conda virtual environment
2. Install CUDA toolkit and build dependencies
3. Configure Conan package management
4. Build C++ extensions
5. Set up Python environment paths

## 🔧 Manual Configuration

### Python Environment Setup
```bash
# Create Conda environment with Tsinghua mirror
conda create -n astroflow \
    --override-channels \
    --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
    --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    python=3.12 conan gcc gxx cuda-toolkit

conda activate astroflow
pip install -r python/requirements.txt

```
### Build C++ Extensions
```bash
# Configure build environment
mkdir build && cd build
conan install .. --output-folder=. --build=missing
source conanbuild.sh

# Build project (Release mode)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Return to project root
cd ..

## 🧪 Verification Tests
```bash
# Validate CUDA extensions
cd build && ctest
```

## 📦 Core Dependencies
- **Compute**: CUDA 11.0+, GCC 9.0+
- **Python**: 3.12+ with NumPy, Matplotlib
- **Build**: CMake 3.20+, Conan 2.0+

## 🤝 Contributing
We welcome contributions through:
1. Issue reporting
2. Pull Requests ([Contribution Guide](CONTRIBUTING.md))
3. Documentation improvements
4. Test case additions

---

🏗️ Maintained by [lintian233](https://github.com/lintian233) | 💬 Discuss via GitHub Issues
