#include "rfimarker_cpu.h"

#include <fstream>
#include <sstream>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
RfiMarkerCPU<T>::RfiMarkerCPU() {
    load_mask("mask.txt");  // 默认尝试 mask.txt，不存在或为空则无坏道
}

template <typename T>
RfiMarkerCPU<T>::RfiMarkerCPU(const char* mask_file) {
    load_mask(mask_file);
}

template <typename T>
void RfiMarkerCPU<T>::load_mask(const char* mask_file) {
    bad_channels_.clear();

    std::ifstream file(mask_file);
    if (!file.is_open()) {
        // 掩码文件不存在 -> 视为无坏道
        return;
    }
    if (file.peek() == std::ifstream::traits_type::eof()) {
        file.close();
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int chan;
        while (ss >> chan) {
            bad_channels_.push_back(chan);
        }
    }
    file.close();
}

template <typename T>
void RfiMarkerCPU<T>::mark_rfi(T* h_data,
                               unsigned int num_channels,
                               unsigned int num_samples)
{
    if (!h_data || bad_channels_.empty() || num_channels == 0 || num_samples == 0) {
        return;
    }

    // 按坏道并行；每条坏道内部时间序列向量化（SIMD）
    #pragma omp parallel for if(bad_channels_.size() > 1) schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(bad_channels_.size()); ++i) {
        const int chan = bad_channels_[static_cast<size_t>(i)];
        if (chan < 0 || static_cast<unsigned int>(chan) >= num_channels) {
            // 非法坏道索引直接跳过
            continue;
        }
        // data[sample * num_channels + chan]
        #pragma omp simd
        for (unsigned int t = 0; t < num_samples; ++t) {
            h_data[static_cast<size_t>(t) * num_channels + static_cast<unsigned int>(chan)] = 0;
        }
    }
}

// 显式实例化定义
template class RfiMarkerCPU<uint8_t>;
template class RfiMarkerCPU<uint16_t>;
template class RfiMarkerCPU<uint32_t>;
