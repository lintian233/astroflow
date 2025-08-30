#include "data.h"
#include "gpucal.h"
#include "iqrm.hpp"
#include "iqrmcuda.h"
#include "marcoutils.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector_types.h>
#include "rfimarker.h"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>
#include <memory>
#include <algorithm>
#include <iostream>
#include <vector>

// #define AF_DEBUG
#ifdef AF_DEBUG
  #include <chrono>
#endif

#ifndef AF_USE_SUBBAND
#define AF_USE_SUBBAND 1
#endif

#ifndef AF_SUBBAND_SIZE_CH
#define AF_SUBBAND_SIZE_CH 32
#endif

#ifndef AF_SUBBAND_NDM0
#define AF_SUBBAND_NDM0 32
#endif

#ifndef AF_SUBBAND_TBLOCK
#define AF_SUBBAND_TBLOCK 81920
#endif

// ------------------------ Logging & Timing helpers ------------------------
#ifdef AF_DEBUG
  #define AF_LOGI(...) do { std::printf("[INFO] "); std::printf(__VA_ARGS__); std::printf("\n"); } while(0)
  #define AF_LOGT(label, msval) do { std::printf("[TIMER] %s : %.3f ms\n", (label), (msval)); } while(0)
  struct AFDebugTimes {
    float total = 0.0f;
    float h2d = 0.0f;
    float time_binning = 0.0f;
    float iqrm_gen = 0.0f;
    float iqrm_mask = 0.0f;
    float zero_dm = 0.0f;
    float mark_file_mask = 0.0f;
    float stage1 = 0.0f;
    float stage2 = 0.0f;
    float host_alloc = 0.0f;
    float d2h = 0.0f;
    float free_ms = 0.0f;
  };
  #define AF_EVENT_PAIR(name) cudaEvent_t name##_start=nullptr, name##_stop=nullptr
  #define AF_EVENT_CREATE(name) do{ cudaEventCreate(&name##_start); cudaEventCreate(&name##_stop);}while(0)
  #define AF_EVENT_DESTROY(name) do{ cudaEventDestroy(name##_start); cudaEventDestroy(name##_stop);}while(0)
  #define AF_EVENT_START(name) do{ cudaEventRecord(name##_start); }while(0)
  #define AF_EVENT_STOP_ACCUM(name, accum) do{ cudaEventRecord(name##_stop); cudaEventSynchronize(name##_stop); float _ms=0.0f; cudaEventElapsedTime(&_ms, name##_start, name##_stop); (accum) += _ms; }while(0)
#else
  #define AF_LOGI(...) do{}while(0)
  #define AF_LOGT(label, msval) do{}while(0)
  struct AFDebugTimes {};
  #define AF_EVENT_PAIR(name)
  #define AF_EVENT_CREATE(name) do{}while(0)
  #define AF_EVENT_DESTROY(name) do{}while(0)
  #define AF_EVENT_START(name) do{}while(0)
  #define AF_EVENT_STOP_ACCUM(name, accum) do{}while(0)
#endif
// -------------------------------------------------------------------------

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while(0)

namespace gpucal {

template<typename T>
__device__ __forceinline__ uint4 load_vector4(const T* ptr) {
    if constexpr (sizeof(T) == 1)      return *reinterpret_cast<const uint4*>(ptr);
    else if constexpr (sizeof(T) == 2) { uint2 v = *reinterpret_cast<const uint2*>(ptr);
      return make_uint4(v.x, v.y, 0, 0); }
    else { uint v = *reinterpret_cast<const uint*>(ptr);
      return make_uint4(v, 0, 0, 0); }
}

template<typename T>
__device__ __forceinline__ uint64_t extract_and_sum_vector4(uint4 vec) {
    if constexpr (sizeof(T) == 1) {
        uint64_t s = 0;
        s += (vec.x & 0xFF) + ((vec.x>>8)&0xFF) + ((vec.x>>16)&0xFF) + ((vec.x>>24)&0xFF);
        s += (vec.y & 0xFF) + ((vec.y>>8)&0xFF) + ((vec.y>>16)&0xFF) + ((vec.y>>24)&0xFF);
        s += (vec.z & 0xFF) + ((vec.z>>8)&0xFF) + ((vec.z>>16)&0xFF) + ((vec.z>>24)&0xFF);
        s += (vec.w & 0xFF) + ((vec.w>>8)&0xFF) + ((vec.w>>16)&0xFF) + ((vec.w>>24)&0xFF);
        return s;
    } else if constexpr (sizeof(T) == 2) {
        uint64_t s = 0;
        s += (vec.x & 0xFFFF) + ((vec.x>>16)&0xFFFF);
        s += (vec.y & 0xFFFF) + ((vec.y>>16)&0xFFFF);
        return s;
    } else {
        return vec.x;
    }
}

template<typename T>
__host__ bool should_use_optimized_kernel(const cudaDeviceProp& p,
                                         size_t nchans, size_t dm_steps,
                                         size_t down_ndata) {
    return (p.major >= 7) && (nchans*dm_steps*down_ndata > 1000000ULL);
}


template <typename T>
__global__ void
dedispersion_shared_memory_kernel_optimized(dedispersion_output_t<T> *output, T *input, int *delay_table,
                                           size_t dm_steps, int time_downsample,
                                           size_t down_ndata, size_t nchans,
                                           size_t chan_start, size_t chan_end,
                                           size_t start, size_t shared_mem_size) {
  const size_t dmidx = blockIdx.y;
  const size_t tidx  = blockIdx.x * blockDim.x + threadIdx.x;
  if (dmidx >= dm_steps || tidx >= down_ndata) return;

  extern __shared__ char sbuf[];
  T* __restrict__ Bloc = reinterpret_cast<T*>(sbuf);

  dedispersion_output_t<T> Sl1=0, Sl2=0, Sl3=0, Sl4=0;
  const size_t Tini = start + blockIdx.x*blockDim.x;

  const size_t Dch  = (chan_end - chan_start + 1);
  const size_t Nch  = min(shared_mem_size / blockDim.x, Dch);
  const int* __restrict__ dptr = &delay_table[dmidx * Dch];

  for (size_t c = 0; c < Dch; c += Nch) {
    const size_t niter = min(Nch, Dch - c);
    const size_t tid = threadIdx.x;

    if (tid < blockDim.x) {
      #pragma unroll 4
      for (size_t off = 0; off < niter; off += 4) {
        if (off + 3 < niter) {
          const size_t ch1 = chan_start + c + off;
          const size_t ch2 = ch1 + 1;
          const size_t ch3 = ch1 + 2;
          const size_t ch4 = ch1 + 3;
          if (ch4 < chan_end) {
            const int d1 = dptr[ch1 - chan_start] / time_downsample;
            const int d2 = dptr[ch2 - chan_start] / time_downsample;
            const int d3 = dptr[ch3 - chan_start] / time_downsample;
            const int d4 = dptr[ch4 - chan_start] / time_downsample;
            const size_t t1 = Tini + tid + d1;
            const size_t t2 = Tini + tid + d2;
            const size_t t3 = Tini + tid + d3;
            const size_t t4 = Tini + tid + d4;
            T v1 = (t1 < down_ndata) ? input[ch1 + t1*nchans] : 0;
            T v2 = (t2 < down_ndata) ? input[ch2 + t2*nchans] : 0;
            T v3 = (t3 < down_ndata) ? input[ch3 + t3*nchans] : 0;
            T v4 = (t4 < down_ndata) ? input[ch4 + t4*nchans] : 0;
            Bloc[(off+0)*blockDim.x + tid] = v1;
            Bloc[(off+1)*blockDim.x + tid] = v2;
            Bloc[(off+2)*blockDim.x + tid] = v3;
            Bloc[(off+3)*blockDim.x + tid] = v4;
          }
        } else {
          for (size_t i = off; i < niter; ++i) {
            const size_t ch = chan_start + c + i;
            if (ch < chan_end) {
              const int d = dptr[ch - chan_start] / time_downsample;
              const size_t t = Tini + tid + d;
              Bloc[i*blockDim.x + tid] = (t < down_ndata) ? input[ch + t*nchans] : 0;
            }
          }
        }
      }
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x) {
      #pragma unroll 8
      for (size_t l = 0; l < niter; l += 8) {
        if (l + 7 < niter) {
          const T* sp = &Bloc[l*blockDim.x + threadIdx.x];
          Sl1 += sp[0*blockDim.x]; Sl2 += sp[1*blockDim.x];
          Sl3 += sp[2*blockDim.x]; Sl4 += sp[3*blockDim.x];
          Sl1 += sp[4*blockDim.x]; Sl2 += sp[5*blockDim.x];
          Sl3 += sp[6*blockDim.x]; Sl4 += sp[7*blockDim.x];
        } else {
          for (size_t i = l; i < niter; ++i) Sl1 += Bloc[i*blockDim.x + threadIdx.x];
        }
      }
    }
    __syncthreads();
  }
  const dedispersion_output_t<T> sum = Sl1+Sl2+Sl3+Sl4;
  output[dmidx*down_ndata + tidx] = sum;
}

template <typename T>
__global__ void
dedispersion_shared_memory_kernel(dedispersion_output_t<T> *output, T *input, int *delay_table,
                                  size_t dm_steps, int time_downsample,
                                  size_t down_ndata, size_t nchans,
                                  size_t chan_start, size_t chan_end,
                                  size_t start, size_t shared_mem_size) {
  size_t dmidx = blockIdx.y;
  size_t tidx = blockIdx.x*blockDim.x + threadIdx.x;
  if (dmidx >= dm_steps || tidx >= down_ndata) return;

  extern __shared__ char sbuf[];
  T* Bloc = reinterpret_cast<T*>(sbuf);
  dedispersion_output_t<T> Sloc = 0;
  size_t Tini = start + blockIdx.x*blockDim.x;

  size_t Dch = (chan_end - chan_start + 1);
  size_t Nch = min(shared_mem_size / blockDim.x, Dch);

  for (size_t c=0; c<Dch; c+=Nch) {
    size_t niter = min(Nch, Dch - c);
    for (size_t off=0; off<niter; ++off) {
      size_t chan = chan_start + c + off;
      if (chan < chan_end && threadIdx.x < blockDim.x) {
        int odelay = delay_table[dmidx * Dch + chan - chan_start];
        size_t d = odelay / time_downsample;
        size_t t = Tini + threadIdx.x + d;
        Bloc[off*blockDim.x + threadIdx.x] = (t < down_ndata) ? input[chan + t*nchans] : 0;
      }
    }
    __syncthreads();

    for (size_t l = 0; l < niter; ++l)
      if (threadIdx.x < blockDim.x) Sloc += Bloc[l*blockDim.x + threadIdx.x];
    __syncthreads();
  }
  output[dmidx*down_ndata + tidx] = Sloc;
}

template <typename T>
__global__ void
dedispersion_kernel_optimized(dedispersion_output_t<T> *output, T *input, int *delay_table,
                             size_t dm_steps, size_t down_ndata, int time_downsample,
                             size_t nchans, size_t chan_start, size_t chan_end,
                             size_t start) {
  const size_t dmidx = blockIdx.y;
  const size_t down_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (dmidx >= dm_steps || down_idx >= down_ndata) return;

  const size_t base_idx = down_idx + start;
  const size_t Dch = chan_end - chan_start;
  const int* __restrict__ dptr = &delay_table[dmidx * Dch];

  dedispersion_output_t<T> s1=0,s2=0,s3=0,s4=0;
  size_t chan = chan_start;
  #pragma unroll 4
  for (; chan + 7 < chan_end; chan += 8) {
    const int d0 = dptr[chan - chan_start] / time_downsample;
    const int d1 = dptr[chan + 1 - chan_start] / time_downsample;
    const int d2 = dptr[chan + 2 - chan_start] / time_downsample;
    const int d3 = dptr[chan + 3 - chan_start] / time_downsample;
    const int d4 = dptr[chan + 4 - chan_start] / time_downsample;
    const int d5 = dptr[chan + 5 - chan_start] / time_downsample;
    const int d6 = dptr[chan + 6 - chan_start] / time_downsample;
    const int d7 = dptr[chan + 7 - chan_start] / time_downsample;

    const size_t t0 = base_idx + d0, t1 = base_idx + d1;
    const size_t t2 = base_idx + d2, t3 = base_idx + d3;
    const size_t t4 = base_idx + d4, t5 = base_idx + d5;
    const size_t t6 = base_idx + d6, t7 = base_idx + d7;

    if (t0 < down_ndata) s1 += input[chan + 0 + t0*nchans];
    if (t1 < down_ndata) s2 += input[chan + 1 + t1*nchans];
    if (t2 < down_ndata) s3 += input[chan + 2 + t2*nchans];
    if (t3 < down_ndata) s4 += input[chan + 3 + t3*nchans];
    if (t4 < down_ndata) s1 += input[chan + 4 + t4*nchans];
    if (t5 < down_ndata) s2 += input[chan + 5 + t5*nchans];
    if (t6 < down_ndata) s3 += input[chan + 6 + t6*nchans];
    if (t7 < down_ndata) s4 += input[chan + 7 + t7*nchans];
  }
  for (; chan < chan_end; ++chan) {
    const int d = dptr[chan - chan_start] / time_downsample;
    const size_t tt = base_idx + d;
    if (tt < down_ndata) s1 += input[chan + tt*nchans];
  }
  output[dmidx*down_ndata + down_idx] = s1+s2+s3+s4;
}

template <typename T>
__global__ void
dedispersion_kernel(dedispersion_output_t<T> *output, T *input, int *delay_table,
                    size_t dm_steps, size_t down_ndata, int time_downsample,
                    size_t nchans, size_t chan_start, size_t chan_end,
                    size_t start) {
  size_t dmidx = blockIdx.y;
  size_t down_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (dmidx >= dm_steps || down_idx >= down_ndata) return;

  size_t base_idx = down_idx + start;
  dedispersion_output_t<T> sum = 0;
  for (size_t chan = chan_start; chan < chan_end; ++chan) {
    int odelay = delay_table[dmidx * (chan_end - chan_start + 1) + chan - chan_start];
    size_t d = odelay / time_downsample;
    size_t tt = base_idx + d;
    if (tt < down_ndata) sum += input[chan + tt*nchans];
  }
  output[dmidx*down_ndata + down_idx] = sum;
}

__global__ void
pre_calculate_dedispersion_kernel(int *delay_table, float dm_low, float dm_high,
                                  float dm_step, size_t chan_start,
                                  size_t chan_end, double *freq_table,
                                  float ref_freq_value, double tsamp) {
  size_t dmidx = blockDim.x*blockIdx.x + threadIdx.x;
  float dm = dm_low + (blockDim.x*blockIdx.x + threadIdx.x) * dm_step;
  if (dm > dm_high) return;
  size_t chan = blockDim.y*blockIdx.y + threadIdx.y + chan_start;
  if (chan > chan_end) return;

  double f = freq_table[chan];
  float ref2  = ref_freq_value * ref_freq_value;
  float f2    = static_cast<float>(f*f);
  float delay = 4148.808f * dm * (1.0f/f2 - 1.0f/ref2);
  delay_table[dmidx * (chan_end - chan_start + 1) + chan - chan_start] =
      static_cast<int>(roundf(delay / tsamp));
}

template <typename T>
__global__ void
time_binning_kernel(T *output, T *input, size_t nchans, size_t ndata,
                   int time_downsample, size_t down_ndata) {
  size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  size_t total = nchans * down_ndata;
  if (idx >= total) return;

  size_t chan = idx % nchans;
  size_t down_idx = idx / nchans;

  size_t t0 = down_idx * static_cast<size_t>(time_downsample);
  size_t t1 = min(t0 + static_cast<size_t>(time_downsample), ndata);

  uint64_t sum = 0;
  constexpr uint64_t vmax = static_cast<uint64_t>(std::numeric_limits<T>::max());
  size_t cnt = t1 - t0;
  for (size_t t = t0; t < t1; ++t)
    sum += static_cast<uint64_t>(input[chan + t*nchans]);

  T result;
  // #define USE_AVERAGE_BINNING
  #define USE_SQRT_COMPRESSION
  #ifdef USE_AVERAGE_BINNING
    result = static_cast<T>(sum / (cnt ? cnt : 1));
  #elif defined(USE_RANDOM_SAMPLING)
    result = input[chan + t0*nchans];
  #elif defined(USE_SQRT_COMPRESSION)
  {
    double s = std::sqrt(static_cast<double>(sum));
    double smax = std::sqrt(static_cast<double>(vmax * (cnt ? cnt : 1)));
    double v = (s / (smax > 0.0 ? smax : 1.0)) * vmax;
    result = static_cast<T>(min(v, static_cast<double>(vmax)));
  }
  #else
    result = static_cast<T>(min(sum, vmax));
  #endif

  output[chan + down_idx*nchans] = result;
}


__global__ void divide_delay_inplace_kernel(int* delay, size_t n, int div) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) delay[i] /= div;
}

// 在 GPU 上并行计算 residual2[dm_steps, NSB]
__global__ void compute_residual2_kernel(
    int* __restrict__ residual2,
    const double* __restrict__ sbfreq,   // [NSB]
    size_t dm_steps, size_t NSB, size_t NDM0,
    float dm_low, float dm_step,
    float ref_freq_value,
    double tsamp, int time_downsample)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = dm_steps * NSB;
  if (idx >= total) return;

  size_t m  = idx / NSB;
  size_t sb = idx % NSB;

  size_t j = m / NDM0;
  float dmj = dm_low + static_cast<float>(j * NDM0) * dm_step; // nominal DM
  float dmm = dm_low + static_cast<float>(m) * dm_step;        // final DM
  float dmdelta = dmm - dmj;

  double f = sbfreq[sb];
  float ref2 = ref_freq_value * ref_freq_value;
  float f2   = static_cast<float>(f * f);
  float delay = 4148.808f * dmdelta * (1.0f/f2 - 1.0f/ref2);

  float bins_f = delay / static_cast<float>(tsamp * time_downsample);
  int   bins   = static_cast<int>(floorf(bins_f + 0.5f)); 
  residual2[idx] = bins;
}


template <typename Tin, typename AccT>
__global__ void subband_stage1_kernel(
    AccT* __restrict__ inter,
    const Tin* __restrict__ input,
    const int* __restrict__ delay1,       
    size_t NDM_nom, size_t NSB, size_t subband_size,
    size_t nchans, size_t chan_start, size_t chan_end,
    size_t time_downsample,                // 已不再使用，仅为接口兼容
    size_t down_ndata,
    size_t t_offset,
    size_t tile1_len,
    size_t Dch)
{
  (void)time_downsample;

  const size_t t  = blockIdx.x*blockDim.x + threadIdx.x;
  const size_t j  = blockIdx.y;      
  const size_t sb = blockIdx.z;    
  if (t >= tile1_len || j >= NDM_nom || sb >= NSB) return;

  size_t ch0 = chan_start + sb*subband_size;
  size_t ch1 = min(ch0 + subband_size, chan_end + 1);

  AccT sum = 0;
  const int* dptr = &delay1[j * Dch];
  for (size_t ch = ch0; ch < ch1; ++ch) {
    int d = dptr[ch - chan_start];      
    size_t tt = t_offset + t + static_cast<size_t>(d);
    if (tt < down_ndata) sum += input[ch + tt*nchans];
  }
  inter[(j*NSB + sb)*tile1_len + t] = sum;
}

template <typename AccT>
__global__ void subband_stage2_kernel(
    AccT* __restrict__ output,
    const AccT* __restrict__ inter,
    const int*  __restrict__ residual2,     
    size_t NDM, size_t NDM0, size_t NSB,
    size_t down_ndata,
    size_t t_offset,
    size_t tile_len,
    size_t tile1_len)
{
  const size_t t   = blockIdx.x*blockDim.x + threadIdx.x;
  const size_t dm  = blockIdx.y;
  if (t >= tile_len || dm >= NDM) return;

  const size_t j = dm / NDM0;    
  AccT sum = 0;

  const int* rptr = &residual2[dm * NSB];
  const AccT* inter_base = &inter[(j * NSB) * tile1_len];

  for (size_t sb=0; sb<NSB; ++sb) {
    int dt = rptr[sb];
    size_t ti = t + static_cast<size_t>(dt);
    if (ti < tile1_len) sum += inter_base[sb*tile1_len + ti];
  }
  output[dm*down_ndata + (t_offset + t)] = sum;
}



// ======================================================================
//                          Main GPU pipelines
// ======================================================================
template <typename T>
dedisperseddata_uint8 dedispered_fil_cuda(Filterbank &fil, float dm_low,
                                    float dm_high, float freq_start,
                                    float freq_end, float dm_step, int ref_freq,
                                    int time_downsample, float t_sample, int target_id,
                                    std::string mask_file, rficonfig rficfg) {
  int device_count; CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count == 0) throw std::runtime_error("No CUDA devices found");

  int device_id = target_id;
  cudaDeviceProp device_prop;
  CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device_id));
  CHECK_CUDA(cudaSetDevice(device_id));

#ifdef AF_DEBUG
  AFDebugTimes tms;
  auto total_start = std::chrono::high_resolution_clock::now();
  AF_EVENT_PAIR(ev_generic);
  AF_EVENT_CREATE(ev_generic);
  AF_EVENT_PAIR(ev_stage1);
  AF_EVENT_CREATE(ev_stage1);
  AF_EVENT_PAIR(ev_stage2);
  AF_EVENT_CREATE(ev_stage2);
#endif

  const float fil_freq_min = fil.frequency_table[0];
  const float fil_freq_max = fil.frequency_table[fil.nchans - 1];

  if (freq_start < fil_freq_min || freq_end > fil_freq_max) {
    char msg[256];
    std::snprintf(msg, sizeof(msg),
             "Frequency range [%.3f-%.3f MHz] out of filterbank range [%.3f-%.3f MHz]",
             freq_start, freq_end, fil_freq_min, fil_freq_max);
    throw std::invalid_argument(msg);
  }

  size_t chan_start = static_cast<size_t>((freq_start - fil_freq_min) /
                          (fil_freq_max - fil_freq_min) * (fil.nchans - 1));
  size_t chan_end   = static_cast<size_t>((freq_end   - fil_freq_min) /
                          (fil_freq_max - fil_freq_min) * (fil.nchans - 1));
  chan_start = std::max<size_t>(0, chan_start);
  chan_end   = std::min<size_t>(fil.nchans - 1, chan_end);
  if (chan_start >= fil.nchans || chan_end >= fil.nchans) {
    char msg[256]; std::snprintf(msg, sizeof(msg),
             "Invalid channel range [%zu-%zu] for %d channels",
             chan_start, chan_end, fil.nchans);
    throw std::invalid_argument(msg);
  }

  const size_t nchans   = fil.nchans;
  const size_t dm_steps = static_cast<size_t>((dm_high - dm_low) / dm_step) + 1;
  const float ref_freq_value = ref_freq ? fil.frequency_table[chan_end]
                                        : fil.frequency_table[chan_start];
  const size_t down_ndata = (fil.ndata + time_downsample - 1) / time_downsample;

  double *d_freq_table;
  CHECK_CUDA(cudaMallocManaged(&d_freq_table, nchans * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_freq_table, fil.frequency_table,
                        nchans*sizeof(double), cudaMemcpyHostToDevice));

#if AF_USE_SUBBAND
  T *h_input = static_cast<T*>(fil.data);
  T *d_input = nullptr;

#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  CHECK_CUDA(cudaMalloc(&d_input, fil.ndata * nchans * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_input, h_input,
                        fil.ndata*nchans*sizeof(T), cudaMemcpyHostToDevice));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.h2d);
#endif

  RfiMarker<T> rfi_marker(mask_file);

  T *d_binned_input = d_input;
  if (time_downsample > 1) {
#ifdef AF_DEBUG
    AF_EVENT_START(ev_generic);
#endif
    CHECK_CUDA(cudaMalloc(&d_binned_input, down_ndata * nchans * sizeof(T)));
    const size_t total = nchans * down_ndata;
    const int TPB = 256;
    const size_t nblk = (total + TPB - 1)/TPB;
    time_binning_kernel<T><<<nblk, TPB>>>(d_binned_input, d_input,
        nchans, fil.ndata, time_downsample, down_ndata);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
#ifdef AF_DEBUG
    AF_EVENT_STOP_ACCUM(ev_generic, tms.time_binning);
#endif
    CHECK_CUDA(cudaFree(d_input));
  }

#ifdef AF_DEBUG
  // 记录 INFO（运行末尾统一输出）
  const bool info_use_iqrm   = rficfg.use_iqrm;
  const bool info_use_zdm    = rficfg.use_zero_dm;
  const bool info_use_mask   = rficfg.use_mask;
#endif

  if (rficfg.use_iqrm) {
#ifdef AF_DEBUG
    auto iqrm_start = std::chrono::high_resolution_clock::now();
#endif
    auto win_masks = iqrm_cuda::rfi_iqrm_gpu<T>(d_binned_input, chan_start, chan_end,
                                                down_ndata, nchans,
                                                fil.tsamp * time_downsample, rficfg);
#ifdef AF_DEBUG
    auto iqrm_end = std::chrono::high_resolution_clock::now();
    tms.iqrm_gen += std::chrono::duration<float, std::milli>(iqrm_end - iqrm_start).count();
    auto rfi_start = std::chrono::high_resolution_clock::now();
#endif
    rfi_marker.mask(d_binned_input, nchans, down_ndata, win_masks);
#ifdef AF_DEBUG
    auto rfi_end = std::chrono::high_resolution_clock::now();
    tms.iqrm_mask += std::chrono::duration<float, std::milli>(rfi_end - rfi_start).count();
#endif
  }

  if (rficfg.use_zero_dm) {
#ifdef AF_DEBUG
    auto z_start = std::chrono::high_resolution_clock::now();
#endif
    rfi_marker.zero_dm_filter(d_binned_input, chan_start, chan_end, down_ndata);
#ifdef AF_DEBUG
    auto z_end = std::chrono::high_resolution_clock::now();
    tms.zero_dm += std::chrono::duration<float, std::milli>(z_end - z_start).count();
#endif
  }
  if (rficfg.use_mask) {
#ifdef AF_DEBUG
    auto m_start = std::chrono::high_resolution_clock::now();
#endif
    rfi_marker.mark_rfi(d_binned_input, nchans, down_ndata);
#ifdef AF_DEBUG
    auto m_end = std::chrono::high_resolution_clock::now();
    tms.mark_file_mask += std::chrono::duration<float, std::milli>(m_end - m_start).count();
#endif
  }

  const size_t Dch = (chan_end - chan_start + 1);
  const size_t NSB = (Dch + AF_SUBBAND_SIZE_CH - 1) / AF_SUBBAND_SIZE_CH;
  const size_t NDM0 = AF_SUBBAND_NDM0;
  const size_t NDM_nom = (dm_steps + NDM0 - 1) / NDM0;

  int *d_delay1 = nullptr;
  CHECK_CUDA(cudaMallocManaged(&d_delay1, NDM_nom * Dch * sizeof(int)));
  {
    float dm_step_coarse = dm_step * static_cast<float>(NDM0);
    float dm_high_coarse = dm_low + (NDM_nom - 1) * dm_step_coarse;

    dim3 bs(64, 16);
    dim3 gs((NDM_nom + bs.x - 1)/bs.x,
            (Dch      + bs.y - 1)/bs.y);
    pre_calculate_dedispersion_kernel<<<gs, bs>>>(
      d_delay1, dm_low, dm_high_coarse, dm_step_coarse,
      chan_start, chan_end, d_freq_table, ref_freq_value, fil.tsamp);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  {
    size_t tot = static_cast<size_t>(NDM_nom) * Dch;
    const int TPB_DIV = 256;
    size_t nblk_div = (tot + TPB_DIV - 1) / TPB_DIV;
    divide_delay_inplace_kernel<<<nblk_div, TPB_DIV>>>(d_delay1, tot, time_downsample);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  std::vector<double> h_sbfreq(NSB, 0.0);
  for (size_t sb=0; sb<NSB; ++sb) {
    size_t ch0 = chan_start + sb*AF_SUBBAND_SIZE_CH;
    size_t ch1 = std::min(ch0 + AF_SUBBAND_SIZE_CH, chan_end + 1);
    size_t mid = (ch0 + ch1 - 1) / 2;
    h_sbfreq[sb] = fil.frequency_table[mid];
  }

  double* d_sbfreq = nullptr;
  CHECK_CUDA(cudaMalloc(&d_sbfreq, NSB * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_sbfreq, h_sbfreq.data(),
                        NSB * sizeof(double), cudaMemcpyHostToDevice));

  int *d_residual2 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_residual2, dm_steps * NSB * sizeof(int)));
  {
    const size_t total = dm_steps * NSB;
    const int TPB = 256;
    const size_t nblk = (total + TPB - 1) / TPB;
    compute_residual2_kernel<<<nblk, TPB>>>(
        d_residual2, d_sbfreq,
        dm_steps, NSB, NDM0,
        dm_low, dm_step, ref_freq_value,
        fil.tsamp, time_downsample);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  int max_residual = 0;
  {
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    int* d_max = nullptr;
    const size_t N = dm_steps * NSB;
    cub::DeviceReduce::Max(d_tmp, tmp_bytes, d_residual2, d_max, N);
    CHECK_CUDA(cudaMalloc(&d_tmp, tmp_bytes));
    CHECK_CUDA(cudaMalloc(&d_max, sizeof(int)));
    cub::DeviceReduce::Max(d_tmp, tmp_bytes, d_residual2, d_max, N);
    CHECK_CUDA(cudaMemcpy(&max_residual, d_max, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaFree(d_max));
  }
  CHECK_CUDA(cudaFree(d_sbfreq));

  dedispersion_output_t<T> *d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_output, dm_steps * down_ndata * sizeof(dedispersion_output_t<T>)));
  CHECK_CUDA(cudaMemset(d_output, 0, dm_steps * down_ndata * sizeof(dedispersion_output_t<T>)));

  const size_t TPB = 256;

  for (size_t t0 = 0; t0 < down_ndata; t0 += AF_SUBBAND_TBLOCK) {
    const size_t tile_len  = std::min(static_cast<size_t>(AF_SUBBAND_TBLOCK), down_ndata - t0);
    const size_t tile1_len = std::min(tile_len + static_cast<size_t>(max_residual), down_ndata - t0);

    dedispersion_output_t<T> *d_inter = nullptr;
    CHECK_CUDA(cudaMalloc(&d_inter, NDM_nom*NSB*tile1_len*sizeof(dedispersion_output_t<T>)));

#ifdef AF_DEBUG
    AF_EVENT_START(ev_stage1);
#endif
    {
      dim3 grid1((tile1_len + TPB - 1)/TPB, NDM_nom, NSB);
      subband_stage1_kernel<T, dedispersion_output_t<T>><<<grid1, TPB>>>(
        d_inter, d_binned_input, d_delay1,
        NDM_nom, NSB, AF_SUBBAND_SIZE_CH,
        nchans, chan_start, chan_end, time_downsample, down_ndata,
        t0, tile1_len, Dch);
      CHECK_CUDA(cudaGetLastError());
    }
#ifdef AF_DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    AF_EVENT_STOP_ACCUM(ev_stage1, tms.stage1);
    AF_EVENT_START(ev_stage2);
#endif
    {
      dim3 grid2((tile_len + TPB - 1)/TPB, dm_steps);
      subband_stage2_kernel<dedispersion_output_t<T>><<<grid2, TPB>>>(
        d_output, d_inter, d_residual2,
        dm_steps, NDM0, NSB,
        down_ndata, t0, tile_len, tile1_len);
      CHECK_CUDA(cudaGetLastError());
    }
#ifdef AF_DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    AF_EVENT_STOP_ACCUM(ev_stage2, tms.stage2);
#endif

    CHECK_CUDA(cudaFree(d_inter));
  }
#ifdef AF_DEBUG
  auto t1 = std::chrono::high_resolution_clock::now();
#endif
  auto dm_array_typed = std::shared_ptr<dedispersion_output_t<T>[]>( new (std::align_val_t{4096})
      dedispersion_output_t<T>[dm_steps * down_ndata](),
      [](dedispersion_output_t<T>* p){ operator delete[](p, std::align_val_t{4096}); });

#ifdef AF_DEBUG
  {
    auto t2 = std::chrono::high_resolution_clock::now();
    tms.host_alloc += std::chrono::duration<float, std::milli>(t2 - t1).count();
  }
  AF_EVENT_START(ev_generic);
#endif
  CHECK_CUDA(cudaMemcpy(dm_array_typed.get(), d_output,
                        dm_steps*down_ndata*sizeof(dedispersion_output_t<T>),
                        cudaMemcpyDeviceToHost));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.d2h);
#endif

#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  if (time_downsample > 1) CHECK_CUDA(cudaFree(d_binned_input));
  else                     CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_delay1));
  CHECK_CUDA(cudaFree(d_residual2));
  CHECK_CUDA(cudaFree(d_freq_table));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.free_ms);
#endif

  DedispersedDataTyped<dedispersion_output_t<T>> typed_result;
  typed_result.dm_times.emplace_back(std::move(dm_array_typed));
  typed_result.dm_low = dm_low;
  typed_result.dm_high = dm_high;
  typed_result.dm_step = dm_step;
  typed_result.tsample = (time_downsample > 1) ? fil.tsamp * time_downsample : fil.tsamp;
  typed_result.filname = fil.filename;
  typed_result.dm_ndata = dm_steps;
  typed_result.downtsample_ndata = down_ndata;
  typed_result.shape = {dm_steps, down_ndata};

  Header temp_header;
  temp_header.tsamp = typed_result.tsample;
  temp_header.filename = fil.filename;

#ifdef AF_DEBUG
  auto total_end = std::chrono::high_resolution_clock::now();
  tms.total = std::chrono::duration<float, std::milli>(total_end - total_start).count();

  AF_LOGI("Using device %d: %s", device_id, device_prop.name);
  AF_LOGI("RFICONFIG use_iqrm=%d, use_zero_dm=%d, use_mask=%d",
          info_use_iqrm, info_use_zdm, info_use_mask);
  AF_LOGT("H2D copy", tms.h2d);
  if (time_downsample > 1) AF_LOGT("Time binning", tms.time_binning);
  if (info_use_iqrm) {
    AF_LOGT("IQRM generate", tms.iqrm_gen);
    AF_LOGT("IQRM mask", tms.iqrm_mask);
  }
  if (info_use_zdm) AF_LOGT("Zero-DM filtering", tms.zero_dm);
  if (info_use_mask) AF_LOGT("Maskfile marking", tms.mark_file_mask);
  AF_LOGT("Stage1 (subband)", tms.stage1);
  AF_LOGT("Stage2 (subband)", tms.stage2);
  AF_LOGT("Host buffer alloc", tms.host_alloc);
  AF_LOGT("D2H copy", tms.d2h);
  AF_LOGT("Free GPU memory", tms.free_ms);
  AF_LOGT("Total", tms.total);

  AF_EVENT_DESTROY(ev_generic);
  AF_EVENT_DESTROY(ev_stage1);
  AF_EVENT_DESTROY(ev_stage2);
#endif

  return preprocess_typed_dedisperseddata_with_slicing<T>(typed_result, temp_header, 1, t_sample);
#else  // ===================== NO SUBBAND BRANCH =========================

#ifdef AF_DEBUG
  AFDebugTimes tms;
  auto total_start = std::chrono::high_resolution_clock::now();
  AF_EVENT_PAIR(ev_generic);
  AF_EVENT_CREATE(ev_generic);
  AF_EVENT_PAIR(ev_stage1); // not used but keep for symmetry
  AF_EVENT_CREATE(ev_stage1);
#endif

  int *d_delay_table;
  CHECK_CUDA(cudaMallocManaged(
      &d_delay_table, dm_steps * (chan_end - chan_start + 1) * sizeof(int)));

  dim3 block_size(64, 16);
  dim3 grid_size((dm_steps + block_size.x - 1) / block_size.x,
                 (chan_end - chan_start + 1 + block_size.y - 1) / block_size.y);

  pre_calculate_dedispersion_kernel<<<grid_size, block_size>>>(
      d_delay_table, dm_low, dm_high, dm_step, chan_start, chan_end,
      d_freq_table, ref_freq_value, fil.tsamp);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  T *d_input;
  T *d_binned_input;
  T *data_ptr = static_cast<T *>(fil.data);

#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  CHECK_CUDA(cudaMalloc(&d_input, fil.ndata * nchans * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_input, data_ptr, fil.ndata * nchans * sizeof(T), cudaMemcpyHostToDevice));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.h2d);
#endif

  if (time_downsample > 1) {
#ifdef AF_DEBUG
    AF_EVENT_START(ev_generic);
#endif
    CHECK_CUDA(cudaMalloc(&d_binned_input, down_ndata * nchans * sizeof(T)));
    const size_t total_elements = nchans * down_ndata;
    const int threads_per_block = 256;
    const size_t blocks_needed = (total_elements + threads_per_block - 1) / threads_per_block;
    time_binning_kernel<T><<<blocks_needed, threads_per_block>>>(
        d_binned_input, d_input, nchans, fil.ndata, time_downsample, down_ndata);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
#ifdef AF_DEBUG
    AF_EVENT_STOP_ACCUM(ev_generic, tms.time_binning);
#endif
    CHECK_CUDA(cudaFree(d_input));
  } else {
    d_binned_input = d_input;
  }

  RfiMarker<T> rfi_marker(mask_file);
#ifdef AF_DEBUG
  auto m_start = std::chrono::high_resolution_clock::now();
#endif
  rfi_marker.mark_rfi(d_binned_input, nchans, down_ndata);
#ifdef AF_DEBUG
  auto m_end = std::chrono::high_resolution_clock::now();
  tms.mark_file_mask += std::chrono::duration<float, std::milli>(m_end - m_start).count();
#endif

  dedispersion_output_t<T> *d_output;
  CHECK_CUDA(cudaMalloc(&d_output, dm_steps * down_ndata * sizeof(dedispersion_output_t<T>)));
  CHECK_CUDA(cudaMemset(d_output, 0, dm_steps * down_ndata * sizeof(dedispersion_output_t<T>)));

  int THREADS_PER_BLOCK = 256;
  dim3 threads(THREADS_PER_BLOCK);
  dim3 grids((down_ndata + threads.x - 1) / threads.x, dm_steps);

  bool use_optimized = should_use_optimized_kernel<T>(device_prop, nchans, dm_steps, down_ndata);
  bool use_shared_memory = true;

  if (use_shared_memory) {
    size_t max_shared_mem = device_prop.sharedMemPerBlock;
    size_t shared_mem_size = std::min(max_shared_mem / sizeof(T),
                                     (chan_end - chan_start + 1) * (size_t)THREADS_PER_BLOCK);
    size_t actual_shared_mem = shared_mem_size * sizeof(T);
    if (use_optimized) {
      dedispersion_shared_memory_kernel_optimized<T><<<grids, threads, actual_shared_mem>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, time_downsample, down_ndata,
          nchans, chan_start, chan_end, 0, shared_mem_size);
    } else {
      dedispersion_shared_memory_kernel<T><<<grids, threads, actual_shared_mem>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, time_downsample, down_ndata,
          nchans, chan_start, chan_end, 0, shared_mem_size);
    }
  } else {
    if (use_optimized) {
      dedispersion_kernel_optimized<T><<<grids, threads>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, down_ndata, time_downsample,
          nchans, chan_start, chan_end, 0);
    } else {
      dedispersion_kernel<T><<<grids, threads>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, down_ndata, time_downsample,
          nchans, chan_start, chan_end, 0);
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  auto dm_array_typed = std::shared_ptr<dedispersion_output_t<T>[]>( new (std::align_val_t{4096})
      dedispersion_output_t<T>[dm_steps * down_ndata](),
      [](dedispersion_output_t<T>* p){ operator delete[](p, std::align_val_t{4096}); });

#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  CHECK_CUDA(cudaMemcpy(dm_array_typed.get(), d_output,
                        dm_steps*down_ndata*sizeof(dedispersion_output_t<T>),
                        cudaMemcpyDeviceToHost));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.d2h);
#endif

#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  if (time_downsample > 1) CHECK_CUDA(cudaFree(d_binned_input));
  else                     CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_delay_table));
  CHECK_CUDA(cudaFree(d_freq_table));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.free_ms);
#endif

  DedispersedDataTyped<dedispersion_output_t<T>> typed_result;
  typed_result.dm_times.emplace_back(std::move(dm_array_typed));
  typed_result.dm_low = dm_low;
  typed_result.dm_high = dm_high;
  typed_result.dm_step = dm_step;
  typed_result.tsample = (time_downsample > 1) ? fil.tsamp * time_downsample : fil.tsamp;
  typed_result.filname = fil.filename;
  typed_result.dm_ndata = dm_steps;
  typed_result.downtsample_ndata = down_ndata;
  typed_result.shape = {dm_steps, down_ndata};

  Header temp_header;
  temp_header.tsamp = typed_result.tsample;
  temp_header.filename = fil.filename;

#ifdef AF_DEBUG
  auto total_end = std::chrono::high_resolution_clock::now();
  tms.total = std::chrono::duration<float, std::milli>(total_end - total_start).count();

  AF_LOGI("Using device %d: %s", device_id, device_prop.name);
  AF_LOGT("H2D copy", tms.h2d);
  if (time_downsample > 1) AF_LOGT("Time binning", tms.time_binning);
  AF_LOGT("Maskfile marking", tms.mark_file_mask);
  AF_LOGT("D2H copy", tms.d2h);
  AF_LOGT("Free GPU memory", tms.free_ms);
  AF_LOGT("Total", tms.total);

  AF_EVENT_DESTROY(ev_generic);
  AF_EVENT_DESTROY(ev_stage1);
#endif

  return preprocess_typed_dedisperseddata_with_slicing<T>(typed_result, temp_header, 1, t_sample);
#endif // AF_USE_SUBBAND
}

template <typename T>
dedisperseddata_uint8 dedisperse_spec(T *data, Header header, float dm_low,
                                float dm_high, float freq_start, float freq_end,
                                float dm_step, int ref_freq,
                                int time_downsample, float t_sample, int target_id,
                                std::string mask_file, rficonfig rficfg) {

  int device_count; CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (!device_count) throw std::runtime_error("No CUDA devices found");

  int device_id = target_id;
  cudaDeviceProp device_prop;
  CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device_id));
  CHECK_CUDA(cudaSetDevice(device_id));

  const size_t nchans = header.nchans;
  std::vector<double> h_freq(nchans);
  for (size_t i=0;i<nchans;++i) h_freq[i] = header.fch1 + i*header.foff;

  float freq_min = h_freq.front();
  float freq_max = h_freq.back();
  if (freq_start < freq_min || freq_end > freq_max) {
    char msg[256];
    std::snprintf(msg, sizeof(msg),
             "Frequency range [%.3f-%.3f MHz] out of spectrum range [%.3f-%.3f MHz]",
             freq_start, freq_end, freq_min, freq_max);
    throw std::invalid_argument(msg);
  }

  size_t chan_start = static_cast<size_t>((freq_start - freq_min) /
                                          (freq_max - freq_min) * (nchans - 1));
  size_t chan_end   = static_cast<size_t>((freq_end   - freq_min) /
                                          (freq_max - freq_min) * (nchans - 1));
  chan_start = std::max<size_t>(0, chan_start);
  chan_end   = std::min<size_t>(nchans - 1, chan_end);
  if (chan_start >= nchans || chan_end >= nchans) {
    char msg[256]; std::snprintf(msg, sizeof(msg),
             "Invalid channel range [%zu-%zu] for %zu channels", chan_start, chan_end, nchans);
    throw std::invalid_argument(msg);
  }

  const size_t dm_steps = static_cast<size_t>((dm_high - dm_low) / dm_step) + 1;
  const float ref_freq_value = ref_freq ? h_freq[chan_end] : h_freq[chan_start];
  const size_t down_ndata = (header.ndata + time_downsample - 1) / time_downsample;

  double *d_freq_table;
  CHECK_CUDA(cudaMallocManaged(&d_freq_table, nchans*sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_freq_table, h_freq.data(),
                        nchans*sizeof(double), cudaMemcpyHostToDevice));

#if AF_USE_SUBBAND
#ifdef AF_DEBUG
  AFDebugTimes tms;
  auto total_start = std::chrono::high_resolution_clock::now();
  AF_EVENT_PAIR(ev_generic);
  AF_EVENT_CREATE(ev_generic);
  AF_EVENT_PAIR(ev_stage1);
  AF_EVENT_CREATE(ev_stage1);
  AF_EVENT_PAIR(ev_stage2);
  AF_EVENT_CREATE(ev_stage2);
#endif

  T *d_input=nullptr;
#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  CHECK_CUDA(cudaMalloc(&d_input, header.ndata*nchans*sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_input, data, header.ndata*nchans*sizeof(T), cudaMemcpyHostToDevice));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.h2d);
#endif

  T *d_binned_input = d_input;
  if (time_downsample > 1) {
#ifdef AF_DEBUG
    AF_EVENT_START(ev_generic);
#endif
    CHECK_CUDA(cudaMalloc(&d_binned_input, down_ndata*nchans*sizeof(T)));
    const size_t total = nchans*down_ndata;
    const int TPB = 256; const size_t nblk = (total + TPB - 1)/TPB;
    time_binning_kernel<T><<<nblk, TPB>>>(
        d_binned_input, d_input, nchans, header.ndata, time_downsample, down_ndata);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
#ifdef AF_DEBUG
    AF_EVENT_STOP_ACCUM(ev_generic, tms.time_binning);
#endif
    CHECK_CUDA(cudaFree(d_input));
  }

#ifdef AF_DEBUG
  const bool info_use_iqrm   = rficfg.use_iqrm;
  const bool info_use_zdm    = rficfg.use_zero_dm;
  const bool info_use_mask   = rficfg.use_mask;
#endif

  RfiMarker<T> rfi_marker(mask_file);
  if (rficfg.use_iqrm) {
#ifdef AF_DEBUG
    auto iqrm_start = std::chrono::high_resolution_clock::now();
#endif
    auto win_masks = iqrm_cuda::rfi_iqrm_gpu<T>(d_binned_input, chan_start, chan_end,
                                                down_ndata, nchans, header.tsamp * time_downsample, rficfg);
#ifdef AF_DEBUG
    auto iqrm_end = std::chrono::high_resolution_clock::now();
    tms.iqrm_gen += std::chrono::duration<float, std::milli>(iqrm_end - iqrm_start).count();
    auto rfi_start = std::chrono::high_resolution_clock::now();
#endif
    rfi_marker.mask(d_binned_input, nchans, down_ndata, win_masks);
#ifdef AF_DEBUG
    auto rfi_end = std::chrono::high_resolution_clock::now();
    tms.iqrm_mask += std::chrono::duration<float, std::milli>(rfi_end - rfi_start).count();
#endif
  }
  
  if (rficfg.use_zero_dm) {
#ifdef AF_DEBUG
    auto z_start = std::chrono::high_resolution_clock::now();
#endif
    rfi_marker.zero_dm_filter(d_binned_input, chan_start, chan_end, down_ndata);
#ifdef AF_DEBUG
    auto z_end = std::chrono::high_resolution_clock::now();
    tms.zero_dm += std::chrono::duration<float, std::milli>(z_end - z_start).count();
#endif
  }
  if (rficfg.use_mask) {
#ifdef AF_DEBUG
    auto m_start = std::chrono::high_resolution_clock::now();
#endif
    rfi_marker.mark_rfi(d_binned_input, nchans, down_ndata);
#ifdef AF_DEBUG
    auto m_end = std::chrono::high_resolution_clock::now();
    tms.mark_file_mask += std::chrono::duration<float, std::milli>(m_end - m_start).count();
#endif
  }

  const size_t Dch  = (chan_end - chan_start + 1);
  const size_t NSB  = (Dch + AF_SUBBAND_SIZE_CH - 1) / AF_SUBBAND_SIZE_CH;
  const size_t NDM0 = AF_SUBBAND_NDM0;
  const size_t NDM_nom = (dm_steps + NDM0 - 1) / NDM0;

  int *d_delay1=nullptr;
  CHECK_CUDA(cudaMallocManaged(&d_delay1, NDM_nom*Dch*sizeof(int)));
  {
    float dm_step_coarse = dm_step * static_cast<float>(NDM0);
    float dm_high_coarse = dm_low + (NDM_nom - 1)*dm_step_coarse;
    dim3 bs(64,16);
    dim3 gs((NDM_nom + bs.x - 1)/bs.x, (Dch + bs.y - 1)/bs.y);
    pre_calculate_dedispersion_kernel<<<gs, bs>>>(
      d_delay1, dm_low, dm_high_coarse, dm_step_coarse,
      chan_start, chan_end, d_freq_table, ref_freq_value, header.tsamp);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  {
    size_t tot = static_cast<size_t>(NDM_nom) * Dch;
    const int TPB_DIV = 256;
    size_t nblk_div = (tot + TPB_DIV - 1) / TPB_DIV;
    divide_delay_inplace_kernel<<<nblk_div, TPB_DIV>>>(d_delay1, tot, time_downsample);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  std::vector<double> h_sbfreq(NSB, 0.0);
  for (size_t sb=0; sb<NSB; ++sb) {
    size_t ch0 = chan_start + sb*AF_SUBBAND_SIZE_CH;
    size_t ch1 = std::min(ch0 + AF_SUBBAND_SIZE_CH, chan_end + 1);
    size_t mid = (ch0 + ch1 - 1)/2;
    h_sbfreq[sb] = h_freq[mid];
  }

  double* d_sbfreq = nullptr;
  CHECK_CUDA(cudaMalloc(&d_sbfreq, NSB * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_sbfreq, h_sbfreq.data(),
                        NSB * sizeof(double), cudaMemcpyHostToDevice));

  int *d_residual2=nullptr;
  CHECK_CUDA(cudaMalloc(&d_residual2, dm_steps * NSB * sizeof(int)));
  {
    const size_t total = dm_steps * NSB;
    const int TPB = 256;
    const size_t nblk = (total + TPB - 1) / TPB;
    compute_residual2_kernel<<<nblk, TPB>>>(
        d_residual2, d_sbfreq,
        dm_steps, NSB, NDM0,
        dm_low, dm_step, ref_freq_value,
        header.tsamp, time_downsample);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  int max_residual = 0;
  {
    void* d_tmp=nullptr; size_t tmp_bytes=0;
    int* d_max=nullptr;
    const size_t N = dm_steps * NSB;
    cub::DeviceReduce::Max(d_tmp, tmp_bytes, d_residual2, d_max, N);
    CHECK_CUDA(cudaMalloc(&d_tmp, tmp_bytes));
    CHECK_CUDA(cudaMalloc(&d_max, sizeof(int)));
    cub::DeviceReduce::Max(d_tmp, tmp_bytes, d_residual2, d_max, N);
    CHECK_CUDA(cudaMemcpy(&max_residual, d_max, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaFree(d_max));
  }
  CHECK_CUDA(cudaFree(d_sbfreq));

  dedispersion_output_t<T> *d_output=nullptr;
  CHECK_CUDA(cudaMalloc(&d_output, dm_steps*down_ndata*sizeof(dedispersion_output_t<T>)));
  CHECK_CUDA(cudaMemset(d_output, 0, dm_steps*down_ndata*sizeof(dedispersion_output_t<T>)));

  const size_t TPB = 256;
  for (size_t t0=0; t0<down_ndata; t0 += AF_SUBBAND_TBLOCK) {
    const size_t tile_len  = std::min(static_cast<size_t>(AF_SUBBAND_TBLOCK), down_ndata - t0);
    const size_t tile1_len = std::min(tile_len + static_cast<size_t>(max_residual), down_ndata - t0);

#ifdef AF_DEBUG
    AF_EVENT_START(ev_stage1);
#endif
    dedispersion_output_t<T>* d_inter=nullptr;
    CHECK_CUDA(cudaMalloc(&d_inter, NDM_nom*NSB*tile1_len*sizeof(dedispersion_output_t<T>)));

    dim3 grid1((tile1_len + TPB - 1)/TPB, NDM_nom, NSB);
    subband_stage1_kernel<T, dedispersion_output_t<T>><<<grid1, TPB>>>(
      d_inter, d_binned_input, d_delay1,
      NDM_nom, NSB, AF_SUBBAND_SIZE_CH,
      nchans, chan_start, chan_end, time_downsample, down_ndata,
      t0, tile1_len, Dch);
    CHECK_CUDA(cudaGetLastError());
#ifdef AF_DEBUG
    CHECK_CUDA(cudaDeviceSynchronize());
    AF_EVENT_STOP_ACCUM(ev_stage1, tms.stage1);
    AF_EVENT_START(ev_stage2);
#endif

    dim3 grid2((tile_len + TPB - 1)/TPB, dm_steps);
    subband_stage2_kernel<dedispersion_output_t<T>><<<grid2, TPB>>>(
      d_output, d_inter, d_residual2,
      dm_steps, NDM0, NSB,
      down_ndata, t0, tile_len, tile1_len);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

#ifdef AF_DEBUG
    AF_EVENT_STOP_ACCUM(ev_stage2, tms.stage2);
#endif
    CHECK_CUDA(cudaFree(d_inter));
  }

  auto dm_array_typed = std::shared_ptr<dedispersion_output_t<T>[]>( new (std::align_val_t{4096})
      dedispersion_output_t<T>[dm_steps * down_ndata](),
      [](dedispersion_output_t<T>* p){ operator delete[](p, std::align_val_t{4096}); });

#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  CHECK_CUDA(cudaMemcpy(dm_array_typed.get(), d_output,
                        dm_steps*down_ndata*sizeof(dedispersion_output_t<T>),
                        cudaMemcpyDeviceToHost));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.d2h);
#endif

#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  if (time_downsample > 1) CHECK_CUDA(cudaFree(d_binned_input));
  else                     CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_delay1));
  CHECK_CUDA(cudaFree(d_residual2));
  CHECK_CUDA(cudaFree(d_freq_table));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.free_ms);
#endif

  DedispersedDataTyped<dedispersion_output_t<T>> typed_result;
  typed_result.dm_times.emplace_back(std::move(dm_array_typed));
  typed_result.dm_low = dm_low;
  typed_result.dm_high = dm_high;
  typed_result.dm_step = dm_step;
  typed_result.tsample = (time_downsample > 1) ? header.tsamp * time_downsample : header.tsamp;
  typed_result.filname = header.filename;
  typed_result.dm_ndata = dm_steps;
  typed_result.downtsample_ndata = down_ndata;
  typed_result.shape = {dm_steps, down_ndata};

  Header updated = header;
  updated.tsamp = typed_result.tsample;

#ifdef AF_DEBUG
  auto total_end = std::chrono::high_resolution_clock::now();
  tms.total = std::chrono::duration<float, std::milli>(total_end - total_start).count();

  AF_LOGI("Using device %d: %s", device_id, device_prop.name);
  AF_LOGI("RFICONFIG use_iqrm=%d, use_zero_dm=%d, use_mask=%d",
          info_use_iqrm, info_use_zdm, info_use_mask);
  AF_LOGT("H2D copy", tms.h2d);
  if (time_downsample > 1) AF_LOGT("Time binning", tms.time_binning);
  if (info_use_iqrm) {
    AF_LOGT("IQRM generate", tms.iqrm_gen);
    AF_LOGT("IQRM mask", tms.iqrm_mask);
  }
  if (info_use_zdm) AF_LOGT("Zero-DM filtering", tms.zero_dm);
  if (info_use_mask) AF_LOGT("Maskfile marking", tms.mark_file_mask);
  AF_LOGT("Stage1 (subband)", tms.stage1);
  AF_LOGT("Stage2 (subband)", tms.stage2);
  AF_LOGT("D2H copy", tms.d2h);
  AF_LOGT("Free GPU memory", tms.free_ms);
  AF_LOGT("Total", tms.total);

  AF_EVENT_DESTROY(ev_generic);
  AF_EVENT_DESTROY(ev_stage1);
  AF_EVENT_DESTROY(ev_stage2);
#endif

  return preprocess_typed_dedisperseddata_with_slicing<T>(typed_result, updated, 1, t_sample);

#else // ===================== NO SUBBAND BRANCH =========================

#ifdef AF_DEBUG
  AFDebugTimes tms;
  auto total_start = std::chrono::high_resolution_clock::now();
  AF_EVENT_PAIR(ev_generic);
  AF_EVENT_CREATE(ev_generic);
#endif

  int *d_delay_table;
  CHECK_CUDA(cudaMallocManaged(
      &d_delay_table, dm_steps * (chan_end - chan_start + 1) * sizeof(int)));

  dim3 block_size(64, 16);
  dim3 grid_size((dm_steps + block_size.x - 1) / block_size.x,
                 (chan_end - chan_start + 1 + block_size.y - 1) / block_size.y);

  pre_calculate_dedispersion_kernel<<<grid_size, block_size>>>(
      d_delay_table, dm_low, dm_high, dm_step, chan_start, chan_end,
      d_freq_table, ref_freq_value, header.tsamp);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  T *d_input; T *d_binned_input;
#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  CHECK_CUDA(cudaMalloc(&d_input, header.ndata*nchans*sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_input, data, header.ndata*nchans*sizeof(T), cudaMemcpyHostToDevice));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.h2d);
#endif

  if (time_downsample > 1) {
#ifdef AF_DEBUG
    AF_EVENT_START(ev_generic);
#endif
    CHECK_CUDA(cudaMalloc(&d_binned_input, down_ndata*nchans*sizeof(T)));
    const size_t total = nchans*down_ndata;
    const int TPB = 256; const size_t nblk = (total + TPB - 1)/TPB;
    time_binning_kernel<T><<<nblk, TPB>>>(
        d_binned_input, d_input, nchans, header.ndata, time_downsample, down_ndata);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
#ifdef AF_DEBUG
    AF_EVENT_STOP_ACCUM(ev_generic, tms.time_binning);
#endif
    CHECK_CUDA(cudaFree(d_input));
  } else d_binned_input = d_input;

  RfiMarker<T> rfi_marker(mask_file);
#ifdef AF_DEBUG
  auto m_start = std::chrono::high_resolution_clock::now();
#endif
  rfi_marker.mark_rfi(d_binned_input, nchans, down_ndata);
#ifdef AF_DEBUG
  auto m_end = std::chrono::high_resolution_clock::now();
  tms.mark_file_mask += std::chrono::duration<float, std::milli>(m_end - m_start).count();
#endif

  dedispersion_output_t<T> *d_output;
  CHECK_CUDA(cudaMalloc(&d_output, dm_steps*down_ndata*sizeof(dedispersion_output_t<T>)));
  CHECK_CUDA(cudaMemset(d_output, 0, dm_steps*down_ndata*sizeof(dedispersion_output_t<T>)));

  int THREADS_PER_BLOCK = 256;
  dim3 threads(THREADS_PER_BLOCK);
  dim3 grids((down_ndata + threads.x - 1) / threads.x, dm_steps);

  bool use_optimized = should_use_optimized_kernel<T>(device_prop, nchans, dm_steps, down_ndata);
  bool use_shared_memory = true;

  if (use_shared_memory) {
    size_t max_shared_mem = device_prop.sharedMemPerBlock;
    size_t shared_mem_size = std::min(max_shared_mem / sizeof(T),
                                      (chan_end - chan_start + 1) * (size_t)THREADS_PER_BLOCK);
    size_t actual_shared_mem = shared_mem_size * sizeof(T);

    if (use_optimized) {
      dedispersion_shared_memory_kernel_optimized<T><<<grids, threads, actual_shared_mem>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, time_downsample, down_ndata,
          nchans, chan_start, chan_end, 0, shared_mem_size);
    } else {
      dedispersion_shared_memory_kernel<T><<<grids, threads, actual_shared_mem>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, time_downsample, down_ndata,
          nchans, chan_start, chan_end, 0, shared_mem_size);
    }
  } else {
    if (use_optimized) {
      dedispersion_kernel_optimized<T><<<grids, threads>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, down_ndata, time_downsample,
          nchans, chan_start, chan_end, 0);
    } else {
      dedispersion_kernel<T><<<grids, threads>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, down_ndata, time_downsample,
          nchans, chan_start, chan_end, 0);
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  auto dm_array_typed = std::shared_ptr<dedispersion_output_t<T>[]>( new (std::align_val_t{4096})
      dedispersion_output_t<T>[dm_steps * down_ndata](),
      [](dedispersion_output_t<T>* p){ operator delete[](p, std::align_val_t{4096}); });

#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  CHECK_CUDA(cudaMemcpy(dm_array_typed.get(), d_output,
                        dm_steps*down_ndata*sizeof(dedispersion_output_t<T>),
                        cudaMemcpyDeviceToHost));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.d2h);
#endif

#ifdef AF_DEBUG
  AF_EVENT_START(ev_generic);
#endif
  if (time_downsample > 1) CHECK_CUDA(cudaFree(d_binned_input));
  else                     CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_delay_table));
  CHECK_CUDA(cudaFree(d_freq_table));
#ifdef AF_DEBUG
  AF_EVENT_STOP_ACCUM(ev_generic, tms.free_ms);
#endif

  DedispersedDataTyped<dedispersion_output_t<T>> typed_result;
  typed_result.dm_times.emplace_back(std::move(dm_array_typed));
  typed_result.dm_low = dm_low;
  typed_result.dm_high = dm_high;
  typed_result.dm_step = dm_step;
  typed_result.tsample = (time_downsample > 1) ? header.tsamp * time_downsample : header.tsamp;
  typed_result.filname = header.filename;
  typed_result.dm_ndata = dm_steps;
  typed_result.downtsample_ndata = down_ndata;
  typed_result.shape = {dm_steps, down_ndata};

  Header updated = header;
  updated.tsamp = typed_result.tsample;

#ifdef AF_DEBUG
  auto total_end = std::chrono::high_resolution_clock::now();
  tms.total = std::chrono::duration<float, std::milli>(total_end - total_start).count();

  AF_LOGI("Using device %d: %s", device_id, device_prop.name);
  AF_LOGT("H2D copy", tms.h2d);
  if (time_downsample > 1) AF_LOGT("Time binning", tms.time_binning);
  AF_LOGT("Maskfile marking", tms.mark_file_mask);
  AF_LOGT("D2H copy", tms.d2h);
  AF_LOGT("Free GPU memory", tms.free_ms);
  AF_LOGT("Total", tms.total);

  AF_EVENT_DESTROY(ev_generic);
#endif

  return preprocess_typed_dedisperseddata_with_slicing<T>(typed_result, updated, 1, t_sample);
#endif // AF_USE_SUBBAND
}


// ---------------------- explicit instantiation ----------------------
template dedisperseddata_uint8
dedispered_fil_cuda<uint8_t>(Filterbank &fil, float dm_low, float dm_high,
                             float freq_start, float freq_end, float dm_step,
                             int ref_freq, int time_downsample, float t_sample,
                             int target_id, std::string mask_file, rficonfig rficfg);

template dedisperseddata_uint8
dedispered_fil_cuda<uint16_t>(Filterbank &fil, float dm_low, float dm_high,
                              float freq_start, float freq_end, float dm_step,
                              int ref_freq, int time_downsample,
                              float t_sample, int target_id, std::string mask_file, rficonfig rficfg);

template dedisperseddata_uint8
dedispered_fil_cuda<uint32_t>(Filterbank &fil, float dm_low, float dm_high,
                              float freq_start, float freq_end, float dm_step,
                              int ref_freq, int time_downsample,
                              float t_sample, int target_id, std::string mask_file, rficonfig rficfg);

template dedisperseddata_uint8
dedisperse_spec<uint8_t>(uint8_t *data, Header header, float dm_low,
                         float dm_high, float freq_start, float freq_end,
                         float dm_step, int ref_freq, int time_downsample,
                         float t_sample, int target_id, std::string mask_file, rficonfig rficfg);

template dedisperseddata_uint8
dedisperse_spec<uint16_t>(uint16_t *data, Header header, float dm_low,
                          float dm_high, float freq_start, float freq_end,
                          float dm_step, int ref_freq, int time_downsample,
                          float t_sample, int target_id, std::string mask_file, rficonfig rficfg);

template dedisperseddata_uint8
dedisperse_spec<uint32_t>(uint32_t *data, Header header, float dm_low,
                          float dm_high, float freq_start, float freq_end,
                          float dm_step, int ref_freq, int time_downsample,
                          float t_sample, int target_id, std::string mask_file, rficonfig rficfg);

} // namespace gpucal
