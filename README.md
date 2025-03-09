# AstroFlow 🌌

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/lintian233/astroflow/build.yml?logo=github)](https://github.com/lintian233/astroflow/actions)
[![C++ Standard](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)](https://isocpp.org/)
[![Python Version](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://www.python.org/)
[![Conda Supported](https://img.shields.io/conda/v/conda-forge/python?logo=anaconda)](https://conda.io/)

A high-performance astronomy data processing framework combining C++ computational efficiency with Python usability, featuring GPU acceleration and automated build configuration.

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
git checkout -b your-feature origin/dev

# Run automated configuration
source configure.sh
```
⚠️ **Note**: The configuration script will:
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
    python=3.12 conan numpy matplotlib gcc gxx cuda-toolkit

conda activate astroflow

# Verify core dependencies
python -c "import numpy, matplotlib; print('Dependency check passed!')"
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
python python/adder.py

# Run visualization demo
python python/visualization/demo_plot.py
```

## 📦 Core Dependencies
- **Compute**: CUDA 11.0+, GCC 9.0+
- **Python**: 3.12+ with NumPy, Matplotlib
- **Math**: Eigen 3.4, BLAS/LAPACK
- **Build**: CMake 3.20+, Conan 2.0+

## 🤝 Contributing
We welcome contributions through:
1. Issue reporting
2. Pull Requests ([Contribution Guide](CONTRIBUTING.md))
3. Documentation improvements
4. Test case additions

## 📜 License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

🏗️ Maintained by [lintian233](https://github.com/lintian233) | 💬 Discuss via GitHub Issues
