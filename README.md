# AstroFlow

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

A high-performance C++ library for radio astronomy signal processing

## ✨ Features

- **Efficient Dedispersion**: Process radio telescope data with optimized algorithms
- **Parallel Processing**: Multi-threaded and OpenMP implementations
- **Flexible Configuration**:
  - Customizable DM ranges and steps
  - Time downsampling support
  - Reference frequency selection
  - Single Pulsar search

## 🚀 Quick Start

```cpp
#include "filterbank.h"
#include "dedispered.hpp"
#include "astrofunc.h"

int main() {
    Filterbank fil("observation.fil");

    //dedispered filterbank
    auto results = dedispered::dedispered_fil_tsample<float>(
        fil, 0.0, 100.0, 0.1, REF_FREQ_TOP, 64, 0.5);

    //Single pulsar search
    fil.info();

    int time_downsample = 4;
    float dm_low = 0;
    float dm_high = 600;
    float freq_start = 1100; // MHz
    float freq_end = 1190;   // MHz
    float dm_step = 1;
    float t_sample = 0.5f;

    single_pulsar_search(fil, dm_low, dm_high, freq_start, freq_end, dm_step,
                       time_downsample, t_sample);
}
```

## 📚 Documentation

### Core Classes

- **Filterbank**: Handles filterbank file I/O and metadata. base on [XLib](https://github.com/ypmen/XLibs/blob/master/src/formats/filterbank.cpp)

- **dedispered**: Namespace containing dedispersion algorithms

### Key Functions

```cpp
template <typename T>
std::vector<std::shared_ptr<T[]>> dedispered_fil_tsample(
    Filterbank &fil, 
    float dm_low, 
    float dm_high,
    float dm_step = 1,
    int ref_freq = REF_FREQ_TOP,
    int time_downsample = 64,
    float t_sample = 0.5);

void single_pulsar_search(Filterbank &fil, float dm_low, float dm_high,
                          float freq_start, float freq_end, float dm_step,
                          int time_downsample, float t_sample); 
```

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lintian233/dedisperedfil.git
   cd dedisperedfil
   ```

2. Init python env:

   ```bash
   #conda env is recommend
   conda create -n astroflow python=3.12 numpy matplotlib
   conda activate astroflow
   
   #Other python environment building methods are being added...
   ```

3. Build with CMake:

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

## 🚧 Roadmap

**Ultimate Goal**: Build a modern, high-performance radio astronomy data processing pipeline centered around CMake, featuring:

- Easy deployment with no manual dependency installation
- Multi-threading (OpenMP) and CUDA support
- Support for multiple astronomical data formats (filterbank, FITS, etc.)
- AI integration for advanced analysis

**Current Phase**: Single-pulse FRB search pipeline

- [x] 1. Filterbank data format reading
- [x] 2. Incoherent dedispersion (tiny FRB search ready)
- [ ] 3. AI image detection for FRB search
- [ ] 4. Boxcar time-slice pulse filtering
- [ ] 5. Candidate generator

## 🤝 Contributing

Contributions are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [lintian233] | [GitHub](https://github.com/lintian233)
