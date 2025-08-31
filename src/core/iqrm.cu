#include <cuda_runtime.h>
#include "iqrmcuda.h"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>


#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        std::cerr << "[CUDA] " << __FILE__ << ":" << __LINE__ \
                  << " " << cudaGetErrorString(_e) << std::endl; \
    } \
} while(0)
#endif

namespace iqrm_cuda {


template <typename T>
__global__ void reduce_sum_sumsq_per_channel_kernel(
    const T* __restrict__ data,   // [nsample * NCHAN]
    unsigned int NCHAN,
    unsigned int t0, unsigned int t1,   // [t0, t1)
    unsigned int chan_start,
    unsigned int Csub,
    double* __restrict__ sum_out,       // [Csub]
    double* __restrict__ sumsq_out      // [Csub]
){
    unsigned int i = blockIdx.x;      // 子带内通道下标 0..Csub-1
    if (i >= Csub) return;

    unsigned int ch = chan_start + i;
    const unsigned int W = t1 - t0;

    // 线程内分段累加
    double s = 0.0;
    double s2 = 0.0;

    // 沿时间轴并行：每 block 处理一个通道，线程跨步累加
    for (unsigned int t = t0 + threadIdx.x; t < t1; t += blockDim.x) {
        double v = static_cast<double>(data[(size_t)t * NCHAN + ch]);
        s  += v;
        s2 += v * v;
    }

    // 共享内存规约
    extern __shared__ unsigned char smem_raw[];
    double* ssum  = reinterpret_cast<double*>(smem_raw);
    double* ssumsq= ssum + blockDim.x;

    ssum[threadIdx.x]   = s;
    ssumsq[threadIdx.x] = s2;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ssum[threadIdx.x]   += ssum[threadIdx.x + stride];
            ssumsq[threadIdx.x] += ssumsq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum_out[i]   = ssum[0];
        sumsq_out[i] = ssumsq[0];
    }
}


__global__ void make_stat_from_sums(
    const double* __restrict__ sum,     // [Csub]
    const double* __restrict__ sumsq,   // [Csub]
    unsigned int W,
    int mode,                           // 0=Mean, 1=Std
    float* __restrict__ x,              // [Csub]
    unsigned int Csub
){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Csub) return;

    double mean = sum[i] / double(W);
    if (mode == 0) {
        x[i] = static_cast<float>(mean);
    } else {
        double var = sumsq[i] / double(W) - mean * mean;
        if (var < 0.0) var = 0.0;
        x[i] = static_cast<float>(sqrt(var));
    }
}


__global__ void make_diff_with_clipping(
    const float* __restrict__ x,  // [Csub]
    int lag,
    float* __restrict__ d,        // [Csub]
    unsigned int Csub
){
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Csub) return;

    if (lag == 0) { d[k] = 0.0f; return; }

    if (lag < 0) {
        // cut = Csub + lag ; edge = x[cut] - x[Csub-1]
        int cut  = (int)Csub + lag;
        float edge = x[cut] - x[Csub - 1];
        if ((int)k < cut) d[k] = x[k] - x[k - lag];   // k + |lag|
        else              d[k] = edge;
    } else {
        // edge = x[lag] - x[0]
        float edge = x[lag] - x[0];
        if ((int)k < lag) d[k] = edge;
        else              d[k] = x[k] - x[k - lag];
    }
}


__global__ void flag_and_count_kernel(
    const float* __restrict__ d,        // [Csub]
    float med,
    float thr,
    int   lag,
    unsigned int Csub,
    // 输出/累加：
    uint8_t* __restrict__ flags_lag,    // [Csub]（本 lag 的布尔掩码）
    int*     __restrict__ recv_cnt,     // [Csub]
    int*     __restrict__ cast_cnt      // [Csub]
){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Csub) return;

    float val = d[i] - med;
    uint8_t f = (val > thr) ? 1u : 0u;
    flags_lag[i] = f;

    if (!f) return;

    int j;
    if (lag < 0) {
        if ((int)i < (int)Csub + lag) j = (int)i - lag;
        else                          j = (int)Csub - 1;
    } else {
        if ((int)i < lag)             j = 0;
        else                          j = (int)i - lag;
    }

    atomicAdd(&recv_cnt[i], 1);
    atomicAdd(&cast_cnt[j], 1);
}


__global__ void merge_lags_to_mask_kernel(
    const uint8_t* __restrict__ flags_all, // [L * Csub]，按 lag 顺序拼接
    const int*     __restrict__ lags,      // [L]
    const int*     __restrict__ recv_cnt,  // [Csub]
    const int*     __restrict__ cast_cnt,  // [Csub]
    unsigned int Csub,
    unsigned int L,
    uint8_t* __restrict__ mask_out         // [Csub]
){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Csub) return;

    int ri = recv_cnt[i];
    if (ri <= 0) { mask_out[i] = 0; return; }

    // 检查是否存在某个 j 对 i 投票，且 cast_cnt[j] < recv_cnt[i]
    for (unsigned int l = 0; l < L; ++l) {
        if (!flags_all[l * Csub + i]) continue;

        int lag = lags[l];
        int j;
        if (lag < 0) {
            if ((int)i < (int)Csub + lag) j = (int)i - lag;
            else                          j = (int)Csub - 1;
        } else {
            if ((int)i < lag)             j = 0;
            else                          j = (int)i - lag;
        }
        if (cast_cnt[j] < ri) { mask_out[i] = 1; return; }
    }
    mask_out[i] = 0;
}



inline std::vector<int> gen_lags_host(int radius, double geofactor) {
    std::vector<int> lags;
    if (radius <= 0) return lags;
    int lag = 1;
    while (lag <= radius) {
        lags.push_back(+lag);
        lags.push_back(-lag);
        int nxt = static_cast<int>(geofactor * lag);
        lag = std::max(nxt, lag + 1);
    }
    return lags;
}


template <typename T>
static std::vector<uint8_t> run_iqrm_one_window_gpu(
    const T* d_data,                   // 设备端数据
    unsigned int NCHAN,
    unsigned int t0, unsigned int t1,  // [t0,t1)
    unsigned int chan_start,
    unsigned int chan_end,
    int mode,                          // 0=Mean, 1=Std
    float radius_frac,
    float nsigma,
    double geofactor,
    cudaStream_t stream = 0
){
    static_assert(std::is_unsigned<T>::value && sizeof(T) <= 4,
                  "T must be uint8/16/32");

    const unsigned int Csub = chan_end - chan_start;
    const unsigned int W    = t1 - t0;
    std::vector<uint8_t> h_mask_sub(Csub, 0u);
    if (Csub == 0 || W == 0) return h_mask_sub;


    double *d_sum = nullptr, *d_sumsq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum,   Csub * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sumsq, Csub * sizeof(double)));

    const int TPB_R = 256;
    dim3 grid_r(Csub, 1, 1);
    size_t shmem_r = TPB_R * 2 * sizeof(double);
    reduce_sum_sumsq_per_channel_kernel<<<grid_r, TPB_R, shmem_r, stream>>>(
        d_data, NCHAN, t0, t1, chan_start, Csub, d_sum, d_sumsq
    );


    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, Csub * sizeof(float)));
    const int TPB_X = 256;
    dim3 grid_x((Csub + TPB_X - 1) / TPB_X);
    make_stat_from_sums<<<grid_x, TPB_X, 0, stream>>>(d_sum, d_sumsq, W, mode, d_x, Csub);


    const int radius = std::max(1, (int)std::floor(radius_frac * (float)Csub));
    std::vector<int> h_lags = gen_lags_host(radius, geofactor);
    const unsigned int L = (unsigned)h_lags.size();


    int    *d_lags     = nullptr;
    float  *d_d        = nullptr;           // 每个 lag 的差分向量
    float  *d_d_sorted = nullptr;           // 排序用
    uint8_t* d_flags_all = nullptr;         // [L * Csub]
    int    *d_recv_cnt = nullptr, *d_cast_cnt = nullptr;
    uint8_t* d_mask_sub = nullptr;          // [Csub]

    if (L > 0) {
        CUDA_CHECK(cudaMalloc(&d_lags,      L * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(d_lags, h_lags.data(), L*sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMalloc(&d_d,         Csub * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_d_sorted,  Csub * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_flags_all, (size_t)L * Csub * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_recv_cnt,  Csub * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cast_cnt,  Csub * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_mask_sub,  Csub * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemsetAsync(d_recv_cnt, 0, Csub * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_cast_cnt, 0, Csub * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_flags_all, 0, (size_t)L * Csub * sizeof(uint8_t), stream));
    } else {

        CUDA_CHECK(cudaFree(d_sum));
        CUDA_CHECK(cudaFree(d_sumsq));
        CUDA_CHECK(cudaFree(d_x));
        return h_mask_sub;
    }

    const int TPB = 256;
    dim3 grid_C((Csub + TPB - 1) / TPB);

    for (unsigned int li = 0; li < L; ++li) {
        int lag;
        CUDA_CHECK(cudaMemcpyAsync(&lag, d_lags + li, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream)); // 获取当前 lag


        make_diff_with_clipping<<<grid_C, TPB, 0, stream>>>(d_x, lag, d_d, Csub);


        CUDA_CHECK(cudaMemcpyAsync(d_d_sorted, d_d, Csub*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        {   // thrust::sort
            thrust::device_ptr<float> td(d_d_sorted);
            thrust::sort(thrust::cuda::par.on(stream), td, td + Csub);
        }


        std::vector<float> tmp(Csub);
        CUDA_CHECK(cudaMemcpyAsync(tmp.data(), d_d_sorted, Csub*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto get_percentile = [&](float q)->float{
            if (Csub == 1) return tmp[0];
            double pos = q * double(Csub - 1);
            size_t lo = (size_t)std::floor(pos);
            size_t hi = (size_t)std::ceil (pos);
            double w  = pos - double(lo);
            return static_cast<float>(tmp[lo] + (tmp[hi] - tmp[lo]) * w);
        };
        float q1  = get_percentile(0.25f);
        float med = get_percentile(0.50f);
        float q3  = get_percentile(0.75f);
        float s   = std::abs((q3 - q1) / 1.349f);

        if (!(s > 0.f) || !std::isfinite(s)) {
            continue;
        }

        float thr = nsigma * s;


        uint8_t* flags_lag = d_flags_all + (size_t)li * Csub;
        flag_and_count_kernel<<<grid_C, TPB, 0, stream>>>(
            d_d, med, thr, lag, Csub, flags_lag, d_recv_cnt, d_cast_cnt
        );
    }

    merge_lags_to_mask_kernel<<<grid_C, TPB, 0, stream>>>(
        d_flags_all, d_lags, d_recv_cnt, d_cast_cnt, Csub, L, d_mask_sub
    );

    CUDA_CHECK(cudaMemcpyAsync(h_mask_sub.data(), d_mask_sub, Csub*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));


    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_sumsq));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_lags));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_d_sorted));
    CUDA_CHECK(cudaFree(d_flags_all));
    CUDA_CHECK(cudaFree(d_recv_cnt));
    CUDA_CHECK(cudaFree(d_cast_cnt));
    CUDA_CHECK(cudaFree(d_mask_sub));

    return h_mask_sub;
}


template <typename T>
std::vector<iqrm_omp::WindowMask> iqrm(
    const T* d_data,              
    unsigned int chan_start,
    unsigned int chan_end,
    double tsamp,
    unsigned int nsample,
    int mode,                     
    cudaStream_t stream = 0
){
    static_assert(std::is_unsigned<T>::value && sizeof(T) <= 4,
                  "T must be uint8/16/32");
    const auto& Cfg = iqrm_omp::cfg();
    const unsigned int NCHAN = Cfg.nchan_total;

    if (!d_data || NCHAN==0 || nsample==0 || chan_start>=chan_end || chan_end>NCHAN)
        return {};

    const unsigned int Csub = chan_end - chan_start;

    const unsigned W = (Cfg.win_sec > 0.0 ? (unsigned)std::floor(Cfg.win_sec / tsamp) : 0u);
    const unsigned H = (Cfg.hop_sec > 0.0 ? (unsigned)std::floor(Cfg.hop_sec / tsamp) : W);

    std::vector<iqrm_omp::WindowMask> out;
    auto push_window = [&](unsigned t0, unsigned t1){
        auto sub = run_iqrm_one_window_gpu(
            d_data, NCHAN, t0, t1, chan_start, chan_end, mode,
            Cfg.radius_frac, Cfg.nsigma, Cfg.geofactor, stream
        );
        
        std::vector<uint8_t> full(NCHAN, 0u);
        if (!sub.empty()) std::copy(sub.begin(), sub.end(), full.begin() + chan_start);
        out.push_back(iqrm_omp::WindowMask{t0, t1, std::move(full)});
    };

    if (W == 0 || W >= nsample) {
        push_window(0, nsample);
    } else {
        unsigned t0 = 0;
        while (t0 + W <= nsample) {
            push_window(t0, t0 + W);
            t0 += (H ? H : W);
        }
        if (Cfg.include_tail && t0 < nsample) {
            unsigned t1 = nsample;
            if (t1 > t0 + 4) push_window(t0, t1);
        }
    }
    return out;
}


template <typename T>
std::vector<iqrm_omp::WindowMask>
rfi_iqrm_gpu_host(const T* h_input,
             size_t chan_start, size_t chan_end,
             size_t nsample, size_t nchan_total,
             double tsamp, const rficonfig& cfg,
             cudaStream_t stream) {

    if (!cfg.use_iqrm) return {};
    iqrm_omp::set_total_channels((unsigned)nchan_total);
    iqrm_omp::set_iqrm_params(cfg.iqrm_cfg.radius_frac,
                              cfg.iqrm_cfg.nsigma,
                              cfg.iqrm_cfg.geofactor);
    iqrm_omp::set_window_seconds(cfg.iqrm_cfg.win_sec,
                                 cfg.iqrm_cfg.hop_sec,
                                 cfg.iqrm_cfg.include_tail);    

    // printf("IQRM CONFIG:" 
    //        " mode=%d radius_frac=%.3f nsigma=%.3f geofactor=%.3f "
    //        " win_sec=%.3f hop_sec=%.3f include_tail=%d\n",
    //        cfg.iqrm_cfg.mode, cfg.iqrm_cfg.radius_frac, cfg.iqrm_cfg.nsigma,
    //        cfg.iqrm_cfg.geofactor, cfg.iqrm_cfg.win_sec, cfg.iqrm_cfg.hop_sec,
    //        (int)cfg.iqrm_cfg.include_tail);
                                     
    T* d_input = nullptr;
    size_t nbytes = (size_t)nsample * nchan_total * sizeof(T);
    CUDA_CHECK(cudaMalloc(&d_input, nbytes));
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, nbytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto masks = iqrm<T>(d_input,
                   (unsigned)chan_start, (unsigned)chan_end,
                   tsamp, (unsigned)nsample, cfg.iqrm_cfg.mode,
                   stream);
    CUDA_CHECK(cudaFree(d_input));
    return masks;
}

template <typename T>
std::vector<iqrm_omp::WindowMask>
rfi_iqrm_gpu(const T* d_input,
             size_t chan_start, size_t chan_end,
             size_t ndata, size_t nchans,
             double tsamp,
             const rficonfig& rficfg,
             cudaStream_t stream)
{
    if (!rficfg.use_iqrm) return {};
    iqrm_omp::set_total_channels((unsigned)nchans);
    iqrm_omp::set_iqrm_params(rficfg.iqrm_cfg.radius_frac,
                              rficfg.iqrm_cfg.nsigma,
                              rficfg.iqrm_cfg.geofactor);
    iqrm_omp::set_window_seconds(rficfg.iqrm_cfg.win_sec,
                                 rficfg.iqrm_cfg.hop_sec,
                                 rficfg.iqrm_cfg.include_tail);

    // printf("IQRM CONFIG:"
    //         " mode=%d radius_frac=%.3f nsigma=%.3f geofactor=%.3f "
    //           " win_sec=%.3f hop_sec=%.3f include_tail=%d\n",
    //           rficfg.iqrm_cfg.mode, rficfg.iqrm_cfg.radius_frac, rficfg.iqrm_cfg.nsigma,
    //           rficfg.iqrm_cfg.geofactor, rficfg.iqrm_cfg.win_sec, rficfg.iqrm_cfg.hop_sec,
    //           (int)rficfg.iqrm_cfg.include_tail);

    return iqrm<T>(d_input,
                   (unsigned)chan_start, (unsigned)chan_end,
                   tsamp, (unsigned)ndata, rficfg.iqrm_cfg.mode,
                   stream);
}

// 显式实例化
template std::vector<iqrm_omp::WindowMask> iqrm<uint8_t >(const uint8_t* , unsigned, unsigned, double, unsigned, int, cudaStream_t);
template std::vector<iqrm_omp::WindowMask> iqrm<uint16_t>(const uint16_t*, unsigned, unsigned, double, unsigned, int, cudaStream_t);
template std::vector<iqrm_omp::WindowMask> iqrm<uint32_t>(const uint32_t*, unsigned, unsigned, double, unsigned, int, cudaStream_t);

template std::vector<iqrm_omp::WindowMask> rfi_iqrm_gpu<uint8_t >(const uint8_t* , size_t, size_t, size_t, size_t, double, const rficonfig&, cudaStream_t);
template std::vector<iqrm_omp::WindowMask> rfi_iqrm_gpu<uint16_t>(const uint16_t*, size_t, size_t, size_t, size_t, double, const rficonfig&, cudaStream_t);
template std::vector<iqrm_omp::WindowMask> rfi_iqrm_gpu<uint32_t>(const uint32_t*, size_t, size_t, size_t, size_t, double, const rficonfig&, cudaStream_t);

template std::vector<iqrm_omp::WindowMask> rfi_iqrm_gpu_host<uint8_t >(const uint8_t*, size_t, size_t, size_t, size_t, double, const rficonfig&, cudaStream_t);
template std::vector<iqrm_omp::WindowMask> rfi_iqrm_gpu_host<uint16_t>(const uint16_t*, size_t, size_t, size_t, size_t, double, const rficonfig&, cudaStream_t);
template std::vector<iqrm_omp::WindowMask> rfi_iqrm_gpu_host<uint32_t>(const uint32_t*, size_t, size_t, size_t, size_t, double, const rficonfig&, cudaStream_t);

} // namespace iqrm_cuda