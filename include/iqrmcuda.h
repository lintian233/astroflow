#pragma once

#ifndef _IQRM_CUDA_H
#define _IQRM_CUDA_H


#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

#include "iqrm.hpp" 
#include "rfi.h" 


namespace iqrm_cuda {

template <typename T>
std::vector<iqrm_omp::WindowMask>
rfi_iqrm_gpu(const T* d_input,
             size_t chan_start, size_t chan_end,
             size_t nsample, size_t nchan_total,
             double tsamp, const rficonfig& cfg,
             cudaStream_t stream = 0);


template <typename T>
std::vector<iqrm_omp::WindowMask>
rfi_iqrm_gpu_host(const T* h_input,
             size_t chan_start, size_t chan_end,
             size_t nsample, size_t nchan_total,
             double tsamp, const rficonfig& cfg,
             cudaStream_t stream = 0);


} // namespace iqrm_cuda

#endif // _IQRM_CUDA_H


