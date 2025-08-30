#include "rfimarker.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>


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

template<typename T, typename AccT>
__global__ void zdm_mean_per_sample_kernel(
    const T* __restrict__ data,
    unsigned int num_channels,
    unsigned int num_samples,
    AccT* __restrict__ mean_per_sample   // [num_samples]
){
    unsigned int t = blockIdx.x;                // one block per sample
    if (t >= num_samples) return;

    // 线程私有累加：仅统计非零样本
    AccT sum = AccT(0);
    unsigned int cnt = 0u;
    for (unsigned int ch = threadIdx.x; ch < num_channels; ch += blockDim.x) {
        size_t idx = static_cast<size_t>(t) * num_channels + ch;
        T v = data[idx];
        if (v != T(0)) {
            sum += static_cast<AccT>(v);
            cnt += 1u;
        }
    }


    extern __shared__ unsigned char smem_raw[];
    AccT*         ssum = reinterpret_cast<AccT*>(smem_raw);
    unsigned int* scnt = reinterpret_cast<unsigned int*>(ssum + blockDim.x);

    ssum[threadIdx.x] = sum;
    scnt[threadIdx.x] = cnt;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            ssum[threadIdx.x] += ssum[threadIdx.x + s];
            scnt[threadIdx.x] += scnt[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        unsigned int nz = scnt[0];
        mean_per_sample[t] = (nz > 0u) ? (ssum[0] / static_cast<AccT>(nz)) : AccT(0);
    }
}

template <typename T>
__global__ void apply_window_masks_kernel(
    T* __restrict__ data,
    unsigned int num_channels,
    const unsigned int* __restrict__ t0s,   // [npairs]
    const unsigned int* __restrict__ t1s,   // [npairs]
    const int*  __restrict__ chans,         // [npairs]
    unsigned int npairs)
{
    unsigned int p = blockIdx.y;
    if (p >= npairs) return;

    const int ch = chans[p];
    if (ch < 0) return;  
    const unsigned int t0 = t0s[p];
    const unsigned int t1 = t1s[p];

    unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int t = t0 + tid;
    while (t < t1) {
        data[static_cast<size_t>(t) * num_channels + static_cast<unsigned int>(ch)] = 0;
        t += stride;
    }
}

template<typename T, typename AccT>
__device__ __forceinline__ T zdm_saturate_sub(T v, AccT delta, bool clip_to_zero){
    AccT r = static_cast<AccT>(v) - delta;
    if (clip_to_zero && r < AccT(0)) r = AccT(0);
    AccT vmax = static_cast<AccT>(std::numeric_limits<T>::max());
    if (r > vmax) r = vmax;
    return static_cast<T>(r);
}

template<typename T, typename AccT>
__global__ void zdm_subtract_kernel(
    T* __restrict__ data,
    unsigned int num_channels,
    unsigned int num_samples,
    const AccT* __restrict__ mean_per_sample,
    float alpha,
    bool clip_to_zero
){
    unsigned int t  = blockIdx.x;                            // time
    unsigned int ch = blockIdx.y * blockDim.x + threadIdx.x; // channel
    if (t >= num_samples || ch >= num_channels) return;

    size_t idx = static_cast<size_t>(t) * num_channels + ch;
    AccT delta = static_cast<AccT>(alpha) * mean_per_sample[t];
    data[idx]  = zdm_saturate_sub<T,AccT>(data[idx], delta, clip_to_zero);
}


template<typename T>
void RfiMarker<T>::zero_dm_filter(T* d_data,
                                  unsigned int num_channels,
                                  unsigned int num_samples,
                                  float alpha,
                                  bool clip_to_zero,
                                  cudaStream_t stream)
{
    if (!d_data || num_channels == 0 || num_samples == 0) return;

    using AccT = float; // 如需更高精度可改 double
    AccT* d_mean = nullptr;
    cudaError_t st = cudaMalloc(&d_mean, sizeof(AccT) * num_samples);
    if (st != cudaSuccess) {
        std::cerr << "cudaMalloc(d_mean) failed: "
                  << cudaGetErrorString(st) << std::endl;
        return;
    }

    const int TPB = 256;

    {
        dim3 grid(num_samples, 1, 1);
        size_t shmem = TPB * (sizeof(AccT) + sizeof(unsigned int));
        zdm_mean_per_sample_kernel<T,AccT><<<grid, TPB, shmem, stream>>>(
            d_data, num_channels, num_samples, d_mean
        );
    }

    {
        unsigned int gy = (num_channels + TPB - 1) / TPB;
        if (gy == 0) gy = 1;
        dim3 grid(num_samples, gy, 1);
        zdm_subtract_kernel<T,AccT><<<grid, TPB, 0, stream>>>(
            d_data, num_channels, num_samples, d_mean, alpha, clip_to_zero
        );
    }

    cudaFree(d_mean);
}


template<typename T, typename AccT>
__global__ void zdm_mean_per_sample_range_kernel(
    const T* __restrict__ data,
    unsigned int num_channels,
    unsigned int num_samples,
    unsigned int chan_start,
    unsigned int chan_end,
    AccT* __restrict__ mean_per_sample   // [num_samples]
){
    unsigned int t = blockIdx.x;  // one block per sample
    if (t >= num_samples) return;

    unsigned int cs = min(chan_start, num_channels);
    unsigned int ce = min(chan_end,   num_channels);
    if (cs >= ce) {
        if (threadIdx.x == 0) mean_per_sample[t] = AccT(0);
        return;
    }

    AccT sum = AccT(0);
    unsigned int cnt = 0u;
    for (unsigned int ch = cs + threadIdx.x; ch < ce; ch += blockDim.x) {
        size_t idx = static_cast<size_t>(t) * num_channels + ch;
        T v = data[idx];
        if (v != T(0)) {
            sum += static_cast<AccT>(v);
            cnt += 1u;
        }
    }

    extern __shared__ unsigned char smem_raw[];
    AccT*         ssum = reinterpret_cast<AccT*>(smem_raw);
    unsigned int* scnt = reinterpret_cast<unsigned int*>(ssum + blockDim.x);

    ssum[threadIdx.x] = sum;
    scnt[threadIdx.x] = cnt;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            ssum[threadIdx.x] += ssum[threadIdx.x + s];
            scnt[threadIdx.x] += scnt[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        unsigned int nz = scnt[0];
        mean_per_sample[t] = (nz > 0u) ? (ssum[0] / static_cast<AccT>(nz)) : AccT(0);
    }
}

template<typename T, typename AccT>
__device__ __forceinline__ T zdm_sat_sub(T v, AccT delta, bool clip_to_zero){
    AccT r = static_cast<AccT>(v) - delta;
    if (clip_to_zero && r < AccT(0)) r = AccT(0);
    AccT vmax = static_cast<AccT>(std::numeric_limits<T>::max());
    if (r > vmax) r = vmax;
    return static_cast<T>(r);
}

template<typename T, typename AccT>
__global__ void zdm_subtract_range_kernel(
    T* __restrict__ data,
    unsigned int num_channels,
    unsigned int num_samples,
    unsigned int chan_start,
    unsigned int chan_end,
    const AccT* __restrict__ mean_per_sample,
    float alpha,
    bool clip_to_zero
){
    unsigned int t  = blockIdx.x;                            
    unsigned int ch = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= num_samples || ch >= num_channels) return;

    unsigned int cs = min(chan_start, num_channels);
    unsigned int ce = min(chan_end,   num_channels);
    if (ch < cs || ch >= ce) return;

    size_t idx = static_cast<size_t>(t) * num_channels + ch;
    AccT delta = static_cast<AccT>(alpha) * mean_per_sample[t];
    data[idx]  = zdm_sat_sub<T,AccT>(data[idx], delta, clip_to_zero);
}


template<typename T>
void RfiMarker<T>::zero_dm_filter(T* d_data,
                                  unsigned int num_channels,
                                  unsigned int num_samples,
                                  unsigned int chan_start,
                                  unsigned int chan_end,
                                  float alpha,
                                  bool clip_to_zero,
                                  cudaStream_t stream)
{
    if (!d_data || num_channels == 0 || num_samples == 0) return;


    if (chan_start > chan_end) { unsigned int tmp = chan_start; chan_start = chan_end; chan_end = tmp; }
    if (chan_start >= num_channels) chan_start = num_channels;
    if (chan_end   >  num_channels) chan_end   = num_channels;
    if (chan_start >= chan_end) return; 

    using AccT = float;
    AccT* d_mean = nullptr;
    cudaError_t st = cudaMalloc(&d_mean, sizeof(AccT) * num_samples);
    if (st != cudaSuccess) {
        std::cerr << "cudaMalloc(d_mean) failed: "
                  << cudaGetErrorString(st) << std::endl;
        return;
    }

    const int TPB = 256;


    {
        dim3 grid(num_samples, 1, 1);
        size_t shmem = TPB * (sizeof(AccT) + sizeof(unsigned int));
        zdm_mean_per_sample_range_kernel<T,AccT><<<grid, TPB, shmem, stream>>>(
            d_data, num_channels, num_samples, chan_start, chan_end, d_mean
        );
    }

    {
        unsigned int gy = (num_channels + TPB - 1) / TPB; 
        if (gy == 0) gy = 1;
        dim3 grid(num_samples, gy, 1);
        zdm_subtract_range_kernel<T,AccT><<<grid, TPB, 0, stream>>>(
            d_data, num_channels, num_samples, chan_start, chan_end, d_mean, alpha, clip_to_zero
        );
    }

    cudaFree(d_mean);
}



template <typename T>
void RfiMarker<T>::mask(T* d_data,
                        unsigned int num_channels,
                        unsigned int num_samples,
                        const std::vector<iqrm_omp::WindowMask>& win_masks,
                        cudaStream_t stream)
{
    if (!d_data || num_channels == 0 || num_samples == 0 || win_masks.empty())
        return;

    // 1) 预扫描：统计三元组总数 & 计算每个窗口的有效时间范围
    size_t total_pairs = 0;
    for (const auto& w : win_masks) {
        if (w.mask.size() != static_cast<size_t>(num_channels)) continue;
        unsigned int t0 = std::min(w.t0, num_samples);
        unsigned int t1 = std::min(w.t1, num_samples);
        if (t0 >= t1) continue;
        // 仅统计 mask==1 的通道数量
        total_pairs += std::count(w.mask.begin(), w.mask.end(), static_cast<uint8_t>(1));
    }
    if (total_pairs == 0) return;

    // 2) 压缩成 (t0,t1,ch) 列表（host）
    std::vector<unsigned int> h_t0s; h_t0s.reserve(total_pairs);
    std::vector<unsigned int> h_t1s; h_t1s.reserve(total_pairs);
    std::vector<int>          h_chs; h_chs.reserve(total_pairs);
    unsigned int max_win_len = 0;

    for (const auto& w : win_masks) {
        if (w.mask.size() != static_cast<size_t>(num_channels)) continue;
        unsigned int t0 = std::min(w.t0, num_samples);
        unsigned int t1 = std::min(w.t1, num_samples);
        if (t0 >= t1) continue;
        unsigned int len = t1 - t0;
        if (len > max_win_len) max_win_len = len;

        for (unsigned int ch = 0; ch < num_channels; ++ch) {
            if (w.mask[ch]) {
                h_t0s.push_back(t0);
                h_t1s.push_back(t1);
                h_chs.push_back(static_cast<int>(ch));
            }
        }
    }
    const unsigned int npairs = static_cast<unsigned int>(h_chs.size());
    if (npairs == 0) return;

    // 3) 设备内存
    unsigned int *d_t0s = nullptr, *d_t1s = nullptr;
    int *d_chs = nullptr;
    cudaError_t st;
    st = cudaMalloc(&d_t0s, npairs * sizeof(unsigned int));
    if (st != cudaSuccess) { std::cerr << "cudaMalloc(d_t0s) failed: " << cudaGetErrorString(st) << "\n"; return; }
    st = cudaMalloc(&d_t1s, npairs * sizeof(unsigned int));
    if (st != cudaSuccess) { std::cerr << "cudaMalloc(d_t1s) failed: " << cudaGetErrorString(st) << "\n"; cudaFree(d_t0s); return; }
    st = cudaMalloc(&d_chs,  npairs * sizeof(int));
    if (st != cudaSuccess) { std::cerr << "cudaMalloc(d_chs) failed: " << cudaGetErrorString(st) << "\n"; cudaFree(d_t0s); cudaFree(d_t1s); return; }

    cudaMemcpyAsync(d_t0s, h_t0s.data(), npairs * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_t1s, h_t1s.data(), npairs * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_chs,  h_chs.data(),  npairs * sizeof(int),          cudaMemcpyHostToDevice, stream);

    // 4) Launch（支持大规模清零；如 npairs 超过 grid.y 上限则分批）
    const int   TPB = 256;
    // 经验：用最大窗口长度来决定 grid.x，保证大窗口也能被 stride 覆盖
    unsigned int gx  = std::max(1u, (max_win_len + TPB - 1) / TPB);
    // 有的设备 gridDim.y 上限 ~ 65535；做个分批以防极端 npairs
    const unsigned int maxY = 65535u;
    for (unsigned int base = 0; base < npairs; base += maxY) {
        unsigned int chunk = std::min(maxY, npairs - base);
        dim3 grid(gx, chunk, 1);
        apply_window_masks_kernel<T><<<grid, TPB, 0, stream>>>(
            d_data, num_channels,
            d_t0s + base, d_t1s + base, d_chs + base,
            chunk
        );
    }

    cudaFree(d_t0s);
    cudaFree(d_t1s);
    cudaFree(d_chs);
}




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
