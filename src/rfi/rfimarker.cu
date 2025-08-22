#include "rfimarker.h"

#include <fstream>
#include <sstream>
#include <iostream>

template <typename T>
__global__ void rfi_zero_channels_kernel(T* __restrict__ data,
                                         unsigned int num_channels,
                                         unsigned int num_samples,
                                         const int* __restrict__ bad_channels,
                                         size_t n_bad)
{
    if (!data || !bad_channels || n_bad == 0) return;

    const size_t bad_idx = blockIdx.y;
    if (bad_idx >= n_bad) return;

    const int chan = bad_channels[bad_idx];
    if (chan < 0 || static_cast<unsigned int>(chan) >= num_channels) return;

    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (; t < num_samples; t += stride) {
        data[static_cast<size_t>(t) * num_channels + static_cast<unsigned int>(chan)] = 0;
    }
}

// ------------------- RfiMarker 实现 -------------------
template <typename T>
RfiMarker<T>::RfiMarker() {
    load_mask("mask.txt");
}

template <typename T>
RfiMarker<T>::RfiMarker(const char* mask_file) {
    load_mask(mask_file);
}

template <typename T>
RfiMarker<T>::~RfiMarker() {
    if (d_bad_channels_) {
        cudaFree(d_bad_channels_);
        d_bad_channels_ = nullptr;
    }
}

template <typename T>
void RfiMarker<T>::upload_bad_channels_to_device() {
    if (d_bad_channels_) {
        cudaFree(d_bad_channels_);
        d_bad_channels_ = nullptr;
    }
    n_bad_ = bad_channels_.size();
    if (n_bad_ == 0) return;

    cudaError_t st = cudaMalloc(&d_bad_channels_, n_bad_ * sizeof(int));
    if (st != cudaSuccess) {
        std::cerr << "cudaMalloc(d_bad_channels_) failed: "
                  << cudaGetErrorString(st) << std::endl;
        d_bad_channels_ = nullptr;
        n_bad_ = 0;
        return;
    }
    st = cudaMemcpy(d_bad_channels_, bad_channels_.data(),
                    n_bad_ * sizeof(int), cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        std::cerr << "cudaMemcpy(H2D) bad_channels failed: "
                  << cudaGetErrorString(st) << std::endl;
        cudaFree(d_bad_channels_);
        d_bad_channels_ = nullptr;
        n_bad_ = 0;
    }
}

template <typename T>
void RfiMarker<T>::load_mask(const char* mask_file) {
    bad_channels_.clear();

    std::ifstream file(mask_file);
    if (!file.is_open()) {
        upload_bad_channels_to_device();
        return;
    }
    if (file.peek() == std::ifstream::traits_type::eof()) {
        file.close();
        upload_bad_channels_to_device();
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

    upload_bad_channels_to_device();
}

template <typename T>
void RfiMarker<T>::mark_rfi(T* d_data,
                            unsigned int num_channels,
                            unsigned int num_samples,
                            cudaStream_t stream)
{
    if (!d_data || n_bad_ == 0 || d_bad_channels_ == nullptr ||
        num_channels == 0 || num_samples == 0) {
        return; 
    }

    constexpr int TPB = 256;
    dim3 block(TPB, 1, 1);
    unsigned int gx = (num_samples + TPB - 1) / TPB;
    if (gx == 0) gx = 1;
    dim3 grid(gx, static_cast<unsigned int>(n_bad_), 1);

    rfi_zero_channels_kernel<T><<<grid, block, 0, stream>>>(
        d_data, num_channels, num_samples, d_bad_channels_, n_bad_);
}

template class RfiMarker<uint8_t>;
template class RfiMarker<uint16_t>;
template class RfiMarker<uint32_t>;
