# AstroFlow

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

## 📦 Contributing

1. Clone the repository:

   ```bash
   git clone https://github.com/lintian233/astroflow.git
   cd astroflow
   git checkout dev
   git checkout -b [new-branch-name]
   ```

2. Init python env:

   ```bash
   #conda env is recommend
   conda create -n astroflow python=3.12 numpy matplotlib
   conda activate astroflow
   
   pip install conan
   #Other python environment building methods are being added...
   ```

3. Build with CMake:

   ```bash
   # you need has cudatoolkit
   mkdir build && cd build
   conan install .. --output-folder=. --build=missing
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make
   ```

4. Test project if install

   ```bash
   python python/adder.py
   ```

Contributions are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [lintian233] | [GitHub](https://github.com/lintian233)
