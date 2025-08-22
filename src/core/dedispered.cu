#include "data.h"
#include "gpucal.h"
#include "marcoutils.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector_types.h>
#include "rfimarker.h"
// timeit
#include <chrono>
#include <cstring>  // for std::memcpy

// cuda atomicAdd

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

namespace gpucal {

// Vectorized memory access helper functions
template<typename T>
__device__ __forceinline__ uint4 load_vector4(const T* ptr) {
    if constexpr (sizeof(T) == 1) {
        return *reinterpret_cast<const uint4*>(ptr);
    } else if constexpr (sizeof(T) == 2) {
        uint2 val = *reinterpret_cast<const uint2*>(ptr);
        return make_uint4(val.x, val.y, 0, 0);
    } else {
        uint val = *reinterpret_cast<const uint*>(ptr);
        return make_uint4(val, 0, 0, 0);
    }
}

template<typename T>
__device__ __forceinline__ uint64_t extract_and_sum_vector4(uint4 vec) {
    if constexpr (sizeof(T) == 1) {
        // Extract 16 bytes and sum them
        uint64_t sum = 0;
        sum += (vec.x & 0xFF) + ((vec.x >> 8) & 0xFF) + ((vec.x >> 16) & 0xFF) + ((vec.x >> 24) & 0xFF);
        sum += (vec.y & 0xFF) + ((vec.y >> 8) & 0xFF) + ((vec.y >> 16) & 0xFF) + ((vec.y >> 24) & 0xFF);
        sum += (vec.z & 0xFF) + ((vec.z >> 8) & 0xFF) + ((vec.z >> 16) & 0xFF) + ((vec.z >> 24) & 0xFF);
        sum += (vec.w & 0xFF) + ((vec.w >> 8) & 0xFF) + ((vec.w >> 16) & 0xFF) + ((vec.w >> 24) & 0xFF);
        return sum;
    } else if constexpr (sizeof(T) == 2) {
        // Extract 8 shorts and sum them
        uint64_t sum = 0;
        sum += (vec.x & 0xFFFF) + ((vec.x >> 16) & 0xFFFF);
        sum += (vec.y & 0xFFFF) + ((vec.y >> 16) & 0xFFFF);
        return sum;
    } else {
        return vec.x;
    }
}

// Helper function to determine optimal kernel configuration
template<typename T>
__host__ bool should_use_optimized_kernel(const cudaDeviceProp& device_prop, 
                                         size_t nchans, size_t dm_steps, 
                                         size_t down_ndata) {
    // Use optimized kernels for newer architectures and larger problem sizes
    if (device_prop.major >= 7) { // Volta and later
        if (nchans * dm_steps * down_ndata > 1000000) { // Large problems benefit more
            return true;
        }
    }
    
    // For smaller problems or older architectures, legacy kernels might be sufficient
    return false;
}

// Optimized shared memory dedispersion kernel with vectorization and loop unrolling
template <typename T>
__global__ void
dedispersion_shared_memory_kernel_optimized(dedispersion_output_t<T> *output, T *input, int *delay_table,
                                           size_t dm_steps, int time_downsample, 
                                           size_t down_ndata, size_t nchans, 
                                           size_t chan_start, size_t chan_end,
                                           size_t start, size_t shared_mem_size) {
  
  // Get thread and block indices
  const size_t dmidx = blockIdx.y;
  const size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (dmidx >= dm_steps || tidx >= down_ndata) {
    return;
  }
  
  // Shared memory buffer with alignment for vectorized access
  extern __shared__ char shared_buffer[];
  T* __restrict__ Bloc = reinterpret_cast<T*>(shared_buffer);
  dedispersion_output_t<T>* __restrict__ accumulator_shared = reinterpret_cast<dedispersion_output_t<T>*>(
      shared_buffer + ((shared_mem_size + 15) & ~15)); // Align to 16 bytes
  
  // Initialize local accumulators with fast bit operations
  dedispersion_output_t<T> Sloc1 = 0, Sloc2 = 0, Sloc3 = 0, Sloc4 = 0;
  
  // Initial time index (already in downsampled space)
  const size_t Tini = start + blockIdx.x * blockDim.x;
  
  // Calculate number of channels per iteration
  const size_t Dch = (chan_end - chan_start + 1);
  constexpr size_t VECTOR_SIZE = sizeof(uint4) / sizeof(T);
  const size_t Nch = min(shared_mem_size / blockDim.x, Dch);
  
  // Precompute delay table offsets
  const int* __restrict__ delay_ptr = &delay_table[dmidx * Dch];
  
  // Process channels in chunks with vectorization
  for (size_t c = 0; c < Dch; c += Nch) {
    const size_t channels_this_iter = min(Nch, Dch - c);
    
    // Vectorized data loading into shared memory
    const size_t tid = threadIdx.x;
    if (tid < blockDim.x) {
      // Load multiple channels per thread with vectorization
      #pragma unroll 4
      for (size_t ch_offset = 0; ch_offset < channels_this_iter; ch_offset += 4) {
        if (ch_offset + 3 < channels_this_iter) {
          // Process 4 channels at once (loop unrolling)
          const size_t chan1 = chan_start + c + ch_offset;
          const size_t chan2 = chan1 + 1;
          const size_t chan3 = chan1 + 2;  
          const size_t chan4 = chan1 + 3;
          
          if (chan4 < chan_end) {
            // Calculate delays for 4 channels
            const int delay1 = delay_ptr[chan1 - chan_start] / time_downsample;
            const int delay2 = delay_ptr[chan2 - chan_start] / time_downsample;
            const int delay3 = delay_ptr[chan3 - chan_start] / time_downsample;
            const int delay4 = delay_ptr[chan4 - chan_start] / time_downsample;
            
            const size_t time_idx1 = Tini + tid + delay1;
            const size_t time_idx2 = Tini + tid + delay2;
            const size_t time_idx3 = Tini + tid + delay3;
            const size_t time_idx4 = Tini + tid + delay4;
            
            // Bounds checking and vectorized loading
            T val1 = (time_idx1 < down_ndata) ? input[chan1 + time_idx1 * nchans] : 0;
            T val2 = (time_idx2 < down_ndata) ? input[chan2 + time_idx2 * nchans] : 0;
            T val3 = (time_idx3 < down_ndata) ? input[chan3 + time_idx3 * nchans] : 0;
            T val4 = (time_idx4 < down_ndata) ? input[chan4 + time_idx4 * nchans] : 0;
            
            // Store in shared memory with coalesced access pattern
            Bloc[(ch_offset + 0) * blockDim.x + tid] = val1;
            Bloc[(ch_offset + 1) * blockDim.x + tid] = val2;
            Bloc[(ch_offset + 2) * blockDim.x + tid] = val3;
            Bloc[(ch_offset + 3) * blockDim.x + tid] = val4;
          }
        } else {
          // Handle remaining channels
          for (size_t ch_idx = ch_offset; ch_idx < channels_this_iter; ++ch_idx) {
            const size_t chan = chan_start + c + ch_idx;
            if (chan < chan_end) {
              const int delay = delay_ptr[chan - chan_start] / time_downsample;
              const size_t time_idx = Tini + tid + delay;
              
              T val = (time_idx < down_ndata) ? input[chan + time_idx * nchans] : 0;
              Bloc[ch_idx * blockDim.x + tid] = val;
            }
          }
        }
      }
    }
    
    // Synchronize threads
    __syncthreads();
    
    // Optimized accumulation with loop unrolling and vectorization
    if (tid < blockDim.x) {
      #pragma unroll 8
      for (size_t l = 0; l < channels_this_iter; l += 8) {
        if (l + 7 < channels_this_iter) {
          // Process 8 channels at once for better utilization
          const T* shared_ptr = &Bloc[l * blockDim.x + tid];
          Sloc1 += shared_ptr[0 * blockDim.x];
          Sloc2 += shared_ptr[1 * blockDim.x];
          Sloc3 += shared_ptr[2 * blockDim.x];
          Sloc4 += shared_ptr[3 * blockDim.x];
          Sloc1 += shared_ptr[4 * blockDim.x];
          Sloc2 += shared_ptr[5 * blockDim.x];
          Sloc3 += shared_ptr[6 * blockDim.x];
          Sloc4 += shared_ptr[7 * blockDim.x];
        } else {
          // Handle remaining channels
          for (size_t ch_idx = l; ch_idx < channels_this_iter; ++ch_idx) {
            Sloc1 += Bloc[ch_idx * blockDim.x + tid];
          }
        }
      }
    }
    
    // Synchronize before next iteration
    __syncthreads();
  }
  
  // Combine all accumulators using bit operations
  const dedispersion_output_t<T> total_sum = Sloc1 + Sloc2 + Sloc3 + Sloc4;
  
  // Store local results into output DM(dm,t) with coalesced write
  if (tidx < down_ndata) {
    output[dmidx * down_ndata + tidx] = total_sum;
  }
}

// Legacy shared memory kernel (kept for compatibility)
template <typename T>
__global__ void
dedispersion_shared_memory_kernel(dedispersion_output_t<T> *output, T *input, int *delay_table,
                                  size_t dm_steps, int time_downsample, 
                                  size_t down_ndata, size_t nchans, 
                                  size_t chan_start, size_t chan_end,
                                  size_t start, size_t shared_mem_size) {
  
  // Get thread and block indices
  size_t dmidx = blockIdx.y;
  size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (dmidx >= dm_steps || tidx >= down_ndata) {
    return;
  }
  
  // Shared memory buffer to store local copy of x(f,t)
  extern __shared__ char shared_buffer[];
  T* Bloc = reinterpret_cast<T*>(shared_buffer);
  
  // Initialize local accumulator
  dedispersion_output_t<T> Sloc = 0;
  
  // Initial time index (already in downsampled space)
  size_t Tini = start + blockIdx.x * blockDim.x;
  
  // Calculate number of channels per iteration
  size_t Dch = (chan_end - chan_start + 1);
  size_t Nch = min(shared_mem_size / blockDim.x, Dch);
  
  // Process channels in chunks
  for (size_t c = 0; c < Dch; c += Nch) {
    size_t channels_this_iter = min(Nch, Dch - c);
    
    // Data segment is stored into shared memory
    for (size_t ch_offset = 0; ch_offset < channels_this_iter; ++ch_offset) {
      size_t chan = chan_start + c + ch_offset;
      if (chan < chan_end && threadIdx.x < blockDim.x) {
        // Calculate the time index with dedispersion delay (in downsampled space)
        int original_delay = delay_table[dmidx * Dch + chan - chan_start];
        size_t delay_in_bins = original_delay / time_downsample; // 转换为降采样后的延迟
        size_t time_idx = Tini + threadIdx.x + delay_in_bins;
        
        // Bounds checking
        if (time_idx < down_ndata) {
          Bloc[ch_offset * blockDim.x + threadIdx.x] = 
            input[chan + time_idx * nchans];
        } else {
          Bloc[ch_offset * blockDim.x + threadIdx.x] = 0;
        }
      }
    }
    
    // Synchronize threads
    __syncthreads();
    
    // Dedisperse local data into accumulators
    for (size_t l = 0; l < channels_this_iter; ++l) {
      if (threadIdx.x < blockDim.x) {
        Sloc += Bloc[l * blockDim.x + threadIdx.x];
      }
    }
    
    // Synchronize before next iteration
    __syncthreads();
  }
  
  // Store local results into output DM(dm,t)
  if (tidx < down_ndata) {
    output[dmidx * down_ndata + tidx] = Sloc;
  }
}

// Optimized global memory dedispersion kernel with vectorization and loop unrolling
template <typename T>
__global__ void
dedispersion_kernel_optimized(dedispersion_output_t<T> *output, T *input, int *delay_table,
                             size_t dm_steps, size_t down_ndata, int time_downsample,
                             size_t nchans, size_t chan_start, size_t chan_end,
                             size_t start) {
  const size_t dmidx = blockIdx.y;
  const size_t down_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (dmidx >= dm_steps || down_idx >= down_ndata)
    return;

  // 在已经降采样的数据上进行去色散
  const size_t base_idx = down_idx + start;
  const size_t Dch = chan_end - chan_start;
  const int* __restrict__ delay_ptr = &delay_table[dmidx * Dch];
  
  // Multiple accumulators for ILP (Instruction Level Parallelism)
  dedispersion_output_t<T> sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
  
  // Process channels with loop unrolling
  size_t chan = chan_start;
  
  // Main unrolled loop - process 8 channels at a time
  #pragma unroll 4
  for (; chan + 7 < chan_end; chan += 8) {
    // Load delays for 8 channels
    const int delay0 = delay_ptr[chan - chan_start] / time_downsample;
    const int delay1 = delay_ptr[chan + 1 - chan_start] / time_downsample;
    const int delay2 = delay_ptr[chan + 2 - chan_start] / time_downsample;
    const int delay3 = delay_ptr[chan + 3 - chan_start] / time_downsample;
    const int delay4 = delay_ptr[chan + 4 - chan_start] / time_downsample;
    const int delay5 = delay_ptr[chan + 5 - chan_start] / time_downsample;
    const int delay6 = delay_ptr[chan + 6 - chan_start] / time_downsample;
    const int delay7 = delay_ptr[chan + 7 - chan_start] / time_downsample;
    
    // Calculate target indices
    const size_t target_idx0 = base_idx + delay0;
    const size_t target_idx1 = base_idx + delay1;
    const size_t target_idx2 = base_idx + delay2;
    const size_t target_idx3 = base_idx + delay3;
    const size_t target_idx4 = base_idx + delay4;
    const size_t target_idx5 = base_idx + delay5;
    const size_t target_idx6 = base_idx + delay6;
    const size_t target_idx7 = base_idx + delay7;
    
    // Bounds checking and accumulation - distribute across multiple accumulators
    if (target_idx0 < down_ndata) sum1 += input[chan + 0 + target_idx0 * nchans];
    if (target_idx1 < down_ndata) sum2 += input[chan + 1 + target_idx1 * nchans];
    if (target_idx2 < down_ndata) sum3 += input[chan + 2 + target_idx2 * nchans];
    if (target_idx3 < down_ndata) sum4 += input[chan + 3 + target_idx3 * nchans];
    if (target_idx4 < down_ndata) sum1 += input[chan + 4 + target_idx4 * nchans];
    if (target_idx5 < down_ndata) sum2 += input[chan + 5 + target_idx5 * nchans];
    if (target_idx6 < down_ndata) sum3 += input[chan + 6 + target_idx6 * nchans];
    if (target_idx7 < down_ndata) sum4 += input[chan + 7 + target_idx7 * nchans];
  }
  
  // Handle remaining channels
  for (; chan < chan_end; ++chan) {
    const int delay = delay_ptr[chan - chan_start] / time_downsample;
    const size_t target_idx = base_idx + delay;
    
    if (target_idx < down_ndata) {
      sum1 += input[chan + target_idx * nchans];
    }
  }
  
  // Combine all accumulators
  const dedispersion_output_t<T> total_sum = sum1 + sum2 + sum3 + sum4;
  output[dmidx * down_ndata + down_idx] = total_sum;
}

// Legacy global memory kernel (kept for compatibility)
template <typename T>
__global__ void
dedispersion_kernel(dedispersion_output_t<T> *output, T *input, int *delay_table,
                    size_t dm_steps, size_t down_ndata, int time_downsample,
                    size_t nchans, size_t chan_start, size_t chan_end,
                    size_t start) {
  size_t dmidx = blockIdx.y;
  size_t down_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dmidx >= dm_steps)
    return;
    
  if (down_idx >= down_ndata)
    return;

  // 在已经降采样的数据上进行去色散
  size_t base_idx = down_idx + start;
  dedispersion_output_t<T> sum = 0;
  for (size_t chan = chan_start; chan < chan_end; ++chan) {
    // 延迟表中的延迟已经按照原始tsamp计算，需要除以time_downsample转换为降采样后的延迟
    int original_delay = delay_table[dmidx * (chan_end - chan_start + 1) + chan - chan_start];
    size_t delay_in_bins = original_delay / time_downsample; // 参数化的time_downsample
    size_t target_idx = base_idx + delay_in_bins;
    
    if (target_idx < down_ndata) {
      sum += input[chan + target_idx * nchans];
    }
  }
  output[dmidx * down_ndata + down_idx] = sum;
}

__global__ void
pre_calculate_dedispersion_kernel(int *delay_table, float dm_low, float dm_high,
                                  float dm_step, size_t chan_start,
                                  size_t chan_end, double *freq_table,
                                  float ref_freq_value, double tsamp) {

  size_t dmidx = blockDim.x * blockIdx.x + threadIdx.x;
  float dm = dm_low + (blockDim.x * blockIdx.x + threadIdx.x) * (dm_step);
  if (dm > dm_high)
    return;
  size_t chan = blockDim.y * blockIdx.y + threadIdx.y + chan_start;
  if (chan > chan_end)
    return;

  double freq = freq_table[chan];
  float ref_2 = ref_freq_value * ref_freq_value;
  float freq_2 = freq * freq;

  float delay = 4148.808f * dm * (1.0f / freq_2 - 1.0f / ref_2);
  delay_table[dmidx * (chan_end - chan_start + 1) + chan - chan_start] =
      static_cast<int>(roundf(delay / tsamp));
}

template <typename T>
__global__ void
time_binning_kernel(T *output, T *input, size_t nchans, size_t ndata, 
                   int time_downsample, size_t down_ndata) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = nchans * down_ndata;
  
  if (idx >= total_elements) {
    return;
  }
  
  size_t chan = idx % nchans;
  size_t down_idx = idx / nchans;
  
  size_t start_time = down_idx * static_cast<size_t>(time_downsample);
  size_t end_time = min(start_time + static_cast<size_t>(time_downsample), ndata);
  
  uint64_t sum = 0;
  constexpr uint64_t max_value = static_cast<uint64_t>(std::numeric_limits<T>::max());
  size_t sample_count = end_time - start_time;
  
  for (size_t t = start_time; t < end_time; ++t) {
    sum += static_cast<uint64_t>(input[chan + t * nchans]);
  }
  
  T result;
  // #define USE_AVERAGE_BINNING
  #define USE_SQRT_COMPRESSION
  #ifdef USE_AVERAGE_BINNING
    // 脉冲星搜索: 平均值，保持周期性信号的相对强度关系
    result = static_cast<T>(sum / sample_count);
  #elif defined(USE_RANDOM_SAMPLING)
    // 单脉冲搜索: 选择第一个样本，保持时间分辨率和峰值特征
    result = input[chan + start_time * nchans];
  #elif defined(USE_SQRT_COMPRESSION)
    // 强信号压缩: 平方根压缩，保持强爆发信号的可检测性
    double sqrt_sum = sqrt(static_cast<double>(sum));
    double sqrt_max_possible = sqrt(static_cast<double>(max_value * sample_count));
    double compressed_value = (sqrt_sum / sqrt_max_possible) * max_value;
    result = static_cast<T>(min(compressed_value, static_cast<double>(max_value)));
  #else
    // 默认策略: 饱和处理
    result = static_cast<T>(min(sum, max_value));
  #endif
  
  output[chan + down_idx * nchans] = result;
}

// [FIRST FUNCTION] Filterbank dedispersion
template <typename T>
dedisperseddata_uint8 dedispered_fil_cuda(Filterbank &fil, float dm_low,
                                    float dm_high, float freq_start,
                                    float freq_end, float dm_step, int ref_freq,
                                    int time_downsample, float t_sample, int target_id,
                                    std::string mask_file) {

  // get all cuda devices
  bool use_shared_memory = true;
  int device_count;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    throw std::runtime_error("No CUDA devices found");
  }

  int device_id = target_id;
  // print device info
  cudaDeviceProp device_prop;
  CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device_id));
  printf("Using device %d: %s\n", device_id, device_prop.name);

  CHECK_CUDA(cudaSetDevice(device_id));

  float fil_freq_min = fil.frequency_table[0];
  float fil_freq_max = fil.frequency_table[fil.nchans - 1];

  if (freq_start < fil_freq_min || freq_end > fil_freq_max) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "Frequency range [%.3f-%.3f MHz] out of filterbank range "
             "[%.3f-%.3f MHz]",
             freq_start, freq_end, fil_freq_min, fil_freq_max);
    throw std::invalid_argument(error_msg);
  }
  
  size_t chan_start =
      static_cast<size_t>((freq_start - fil_freq_min) /
                          (fil_freq_max - fil_freq_min) * (fil.nchans - 1));
  size_t chan_end =
      static_cast<size_t>((freq_end - fil_freq_min) /
                          (fil_freq_max - fil_freq_min) * (fil.nchans - 1));

  chan_start = std::max(static_cast<size_t>(0), chan_start);
  chan_end = std::min(static_cast<size_t>(fil.nchans - 1), chan_end);

  if (chan_start >= fil.nchans || chan_end >= fil.nchans) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "Invalid channel range [%zu-%zu] for %d channels", chan_start,
             chan_end, fil.nchans);
    throw std::invalid_argument(error_msg);
  }

  const size_t nchans = fil.nchans;
  const size_t dm_steps = static_cast<size_t>((dm_high - dm_low) / dm_step) + 1;
  const float ref_freq_value = ref_freq ? fil.frequency_table[chan_end]
                                        : fil.frequency_table[chan_start];

  // Calculate the full downsampled time dimensions
  const size_t down_ndata = (fil.ndata + time_downsample - 1) / time_downsample;

  int *d_delay_table;
  CHECK_CUDA(cudaMallocManaged(
      &d_delay_table, dm_steps * (chan_end - chan_start + 1) * sizeof(int)));

  double *d_freq_table;
  CHECK_CUDA(cudaMallocManaged(&d_freq_table, nchans * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_freq_table, fil.frequency_table,
                        nchans * sizeof(double), cudaMemcpyHostToDevice));

  dim3 block_size(64, 16);
  dim3 grid_size((dm_steps + block_size.x - 1) / block_size.x,
                 (chan_end - chan_start + 1 + block_size.y - 1) / block_size.y);

  pre_calculate_dedispersion_kernel<<<grid_size, block_size>>>(
      d_delay_table, dm_low, dm_high, dm_step, chan_start, chan_end,
      d_freq_table, ref_freq_value, fil.tsamp);

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  T *d_input;
  T *d_binned_input; // 存储分bin后的数据
  T *data = static_cast<T *>(fil.data);

  CHECK_CUDA(cudaMalloc(&d_input, fil.ndata * nchans * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_input, data, fil.ndata * nchans * sizeof(T),
                        cudaMemcpyHostToDevice));

  if (time_downsample > 1) {
    // 需要进行时间分bin降采样
    CHECK_CUDA(cudaMalloc(&d_binned_input, down_ndata * nchans * sizeof(T)));
    
    printf("Performing time binning: %zu -> %zu time samples (factor %d)\n", 
           fil.ndata, down_ndata, time_downsample);
    
    // 使用1D grid配置来避免grid大小限制问题
    const size_t total_elements = nchans * down_ndata;
    const int threads_per_block = 256;
    const size_t blocks_needed = (total_elements + threads_per_block - 1) / threads_per_block;
    
    printf("Binning kernel config: %zu total elements, %zu blocks, %d threads per block\n", 
           total_elements, blocks_needed, threads_per_block);
    
    auto binning_start = std::chrono::high_resolution_clock::now();
    

    time_binning_kernel<T><<<blocks_needed, threads_per_block>>>(
        d_binned_input, d_input, nchans, fil.ndata, time_downsample, down_ndata);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    auto binning_end = std::chrono::high_resolution_clock::now();
    auto binning_duration = std::chrono::duration_cast<std::chrono::milliseconds>(binning_end - binning_start);
    printf("Time binning completed in %lld ms\n", binning_duration.count());
    
    CHECK_CUDA(cudaFree(d_input));
  } else {
    // 不需要分bin，直接使用原始数据
    printf("No time binning needed (factor = 1)\n");
    d_binned_input = d_input;
  }

  printf("Processing full data: DM steps = %zu, Time samples = %zu\n", dm_steps, down_ndata);

  dedispersion_output_t<T> *d_output;
  CHECK_CUDA(cudaMalloc(&d_output, dm_steps * down_ndata * sizeof(dedispersion_output_t<T>)));
  CHECK_CUDA(cudaMemset(d_output, 0, dm_steps * down_ndata * sizeof(dedispersion_output_t<T>)));

  RfiMarker<T> rfi_marker(mask_file);
  rfi_marker.mark_rfi(d_binned_input, nchans, down_ndata);

  int THREADS_PER_BLOCK = 256;
  dim3 threads(THREADS_PER_BLOCK);
  dim3 grids((down_ndata + threads.x - 1) / threads.x, dm_steps);
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // 决定是否使用优化内核
  bool use_optimized = should_use_optimized_kernel<T>(device_prop, nchans, dm_steps, down_ndata);
  
  if (use_shared_memory) {
    // Calculate shared memory size needed
    size_t max_shared_mem = device_prop.sharedMemPerBlock;
    size_t shared_mem_size = std::min(max_shared_mem / sizeof(T), 
                                     (chan_end - chan_start + 1) * THREADS_PER_BLOCK);
    
    // Ensure we don't exceed shared memory limits
    size_t actual_shared_mem = shared_mem_size * sizeof(T);
    
    if (use_optimized) {
      printf("Using optimized shared memory kernel with %zu bytes of shared memory\n", actual_shared_mem);
      dedispersion_shared_memory_kernel_optimized<T><<<grids, threads, actual_shared_mem>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, time_downsample, down_ndata,
          nchans, chan_start, chan_end, 0, shared_mem_size);
    } else {
      printf("Using legacy shared memory kernel with %zu bytes of shared memory\n", actual_shared_mem);
      dedispersion_shared_memory_kernel<T><<<grids, threads, actual_shared_mem>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, time_downsample, down_ndata,
          nchans, chan_start, chan_end, 0, shared_mem_size);
    }
  } else {
    if (use_optimized) {
      printf("Using optimized global memory kernel\n");
      dedispersion_kernel_optimized<T><<<grids, threads>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, down_ndata, time_downsample,
          nchans, chan_start, chan_end, 0);
    } else {
      printf("Using legacy global memory kernel\n");
      dedispersion_kernel<T><<<grids, threads>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, down_ndata, time_downsample,
          nchans, chan_start, chan_end, 0);
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  printf("Dedispersion completed in %lld ms\n", duration.count());

  // Optimized memory copy - single large transfer to maximize PCIe bandwidth
  auto copy_start = std::chrono::high_resolution_clock::now();
  
  // Create aligned array directly for final storage
  auto dm_array_typed = std::shared_ptr<dedispersion_output_t<T>[]>(
      new (std::align_val_t{4096}) dedispersion_output_t<T>[dm_steps * down_ndata](),
      [](dedispersion_output_t<T> *p) { operator delete[](p, std::align_val_t{4096}); });
  
  // Single large transfer to maximize PCIe bandwidth utilization
  const size_t total_bytes = dm_steps * down_ndata * sizeof(dedispersion_output_t<T>);
  printf("Using single large memory copy for %zu MB to maximize PCIe bandwidth\n", total_bytes / (1024 * 1024));
  
  CHECK_CUDA(cudaMemcpy(dm_array_typed.get(), d_output,
                        total_bytes, cudaMemcpyDeviceToHost));
  
  auto copy_end = std::chrono::high_resolution_clock::now();
  auto copy_duration = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - copy_start);
  printf("Memory copy completed in %lld ms (%.2f GB/s)\n", 
         copy_duration.count(), 
         (total_bytes / 1024.0 / 1024.0 / 1024.0) / (copy_duration.count() / 1000.0));

  // 清理GPU资源 - [FIRST FUNCTION fil.tsamp context]
  if (time_downsample > 1) {
    CHECK_CUDA(cudaFree(d_binned_input));
  } else {
    CHECK_CUDA(cudaFree(d_input)); // d_binned_input == d_input when no binning
  }
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_delay_table));
  CHECK_CUDA(cudaFree(d_freq_table));

  // Create typed dedispersion data structure - no additional copy needed
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

  printf("Full dedispersion completed. Now applying type-safe preprocessing with slicing...\n");
  // 对于Filterbank，需要构造Header结构
  Header temp_header;
  temp_header.tsamp = (time_downsample > 1) ? fil.tsamp * time_downsample : fil.tsamp;
  temp_header.filename = fil.filename;
  return preprocess_typed_dedisperseddata_with_slicing<T>(typed_result, temp_header, 1, t_sample);
}

template <typename T>
dedisperseddata_uint8 dedisperse_spec(T *data, Header header, float dm_low,
                                float dm_high, float freq_start, float freq_end,
                                float dm_step, int ref_freq,
                                int time_downsample, float t_sample, int target_id,
                                std::string mask_file) { 
  // get all cuda devices
  bool use_shared_memory = true; // 是否使用共享内存

  int device_count;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    throw std::runtime_error("No CUDA devices found");
  }

  int device_id = target_id;
  // print device info
  cudaDeviceProp device_prop;
  CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device_id));
  printf("Using device %d: %s\n", device_id, device_prop.name);

  CHECK_CUDA(cudaSetDevice(device_id));
  
  // Calculate frequency table from header info
  const size_t nchans = header.nchans;
  std::vector<double> frequency_table(nchans);
  float fch1 = header.fch1;
  float foff = header.foff;

  for (size_t i = 0; i < nchans; ++i) {
    frequency_table[i] = fch1 + i * foff;
  }

  float freq_min = frequency_table[0];
  float freq_max = frequency_table[nchans - 1];

  // Validate parameters
  if (freq_start < freq_min || freq_end > freq_max) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "Frequency range [%.3f-%.3f MHz] out of spectrum range "
             "[%.3f-%.3f MHz]",
             freq_start, freq_end, freq_min, freq_max);
    throw std::invalid_argument(error_msg);
  }

  size_t chan_start = static_cast<size_t>((freq_start - freq_min) /
                                          (freq_max - freq_min) * (nchans - 1));
  size_t chan_end = static_cast<size_t>((freq_end - freq_min) /
                                        (freq_max - freq_min) * (nchans - 1));

  chan_start = std::max(static_cast<size_t>(0), chan_start);
  chan_end = std::min(static_cast<size_t>(nchans) - 1, chan_end);

  if (chan_start >= nchans || chan_end >= nchans) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "Invalid channel range [%zu-%zu] for %zu channels", chan_start,
             chan_end, nchans);
    throw std::invalid_argument(error_msg);
  }

  const size_t dm_steps = static_cast<size_t>((dm_high - dm_low) / dm_step) + 1;
  const float ref_freq_value =
      ref_freq ? frequency_table[chan_end] : frequency_table[chan_start];

  // Calculate the full downsampled time dimensions
  const size_t down_ndata = (header.ndata + time_downsample - 1) / time_downsample;

  // Allocate and initialize delay table on GPU
  int *d_delay_table;
  CHECK_CUDA(cudaMallocManaged(
      &d_delay_table, dm_steps * (chan_end - chan_start + 1) * sizeof(int)));

  double *d_freq_table;
  CHECK_CUDA(cudaMallocManaged(&d_freq_table, nchans * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_freq_table, frequency_table.data(),
                        nchans * sizeof(double), cudaMemcpyHostToDevice));

  // Calculate dedispersion delay table
  dim3 block_size(64, 16);
  dim3 grid_size((dm_steps + block_size.x - 1) / block_size.x,
                 (chan_end - chan_start + 1 + block_size.y - 1) / block_size.y);

  pre_calculate_dedispersion_kernel<<<grid_size, block_size>>>(
      d_delay_table, dm_low, dm_high, dm_step, chan_start, chan_end,
      d_freq_table, ref_freq_value, header.tsamp);

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());


  T *d_input;
  T *d_binned_input; // 存储分bin后的数据
  CHECK_CUDA(cudaMalloc(&d_input, header.ndata * nchans * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_input, data, header.ndata * nchans * sizeof(T),
                        cudaMemcpyHostToDevice));

  if (time_downsample > 1) {
    // 需要进行时间分bin降采样
    CHECK_CUDA(cudaMalloc(&d_binned_input, down_ndata * nchans * sizeof(T)));
    
    printf("Performing time binning: %zu -> %zu time samples (factor %d)\n", 
           header.ndata, down_ndata, time_downsample);
    
    // 使用1D grid配置来避免grid大小限制问题
    const size_t total_elements = nchans * down_ndata;
    const int threads_per_block = 256;
    const size_t blocks_needed = (total_elements + threads_per_block - 1) / threads_per_block;
    
    printf("Binning kernel config: %zu total elements, %zu blocks, %d threads per block\n", 
           total_elements, blocks_needed, threads_per_block);
    
    auto binning_start = std::chrono::high_resolution_clock::now();
    
    time_binning_kernel<T><<<blocks_needed, threads_per_block>>>(
          d_binned_input, d_input, nchans, header.ndata, time_downsample, down_ndata);

    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    auto binning_end = std::chrono::high_resolution_clock::now();
    auto binning_duration = std::chrono::duration_cast<std::chrono::milliseconds>(binning_end - binning_start);
    printf("Time binning completed in %lld ms\n", binning_duration.count());
    
    // 释放原始输入数据
    CHECK_CUDA(cudaFree(d_input));
  } else {
    // 不需要分bin，直接使用原始数据
    printf("No time binning needed (factor = 1)\n");
    d_binned_input = d_input;
  }

  printf("Processing full data: DM steps = %zu, Time samples = %zu\n", dm_steps, down_ndata);

  RfiMarker<T> rfi_marker(mask_file);
  rfi_marker.mark_rfi(d_binned_input, nchans, down_ndata);

  dedispersion_output_t<T> *d_output;
  CHECK_CUDA(cudaMalloc(&d_output, dm_steps * down_ndata * sizeof(dedispersion_output_t<T>)));
  CHECK_CUDA(cudaMemset(d_output, 0, dm_steps * down_ndata * sizeof(dedispersion_output_t<T>)));

  int THREADS_PER_BLOCK = 256;
  dim3 threads(THREADS_PER_BLOCK);
  dim3 grids((down_ndata + threads.x - 1) / threads.x, dm_steps);

  // 决定是否使用优化内核
  bool use_optimized = should_use_optimized_kernel<T>(device_prop, nchans, dm_steps, down_ndata);
  auto start_time = std::chrono::high_resolution_clock::now();
  if (use_shared_memory) {
    // Calculate shared memory size needed
    size_t max_shared_mem = device_prop.sharedMemPerBlock;
    size_t shared_mem_size = std::min(max_shared_mem / sizeof(T), 
                                     (chan_end - chan_start + 1) * THREADS_PER_BLOCK);
    
    // Ensure we don't exceed shared memory limits
    size_t actual_shared_mem = shared_mem_size * sizeof(T);
    
    if (use_optimized) {
      printf("Using optimized shared memory kernel with %zu bytes of shared memory\n", actual_shared_mem);
      dedispersion_shared_memory_kernel_optimized<T><<<grids, threads, actual_shared_mem>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, time_downsample, down_ndata,
          nchans, chan_start, chan_end, 0, shared_mem_size);
    } else {
      printf("Using legacy shared memory kernel with %zu bytes of shared memory\n", actual_shared_mem);
      dedispersion_shared_memory_kernel<T><<<grids, threads, actual_shared_mem>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, time_downsample, down_ndata,
          nchans, chan_start, chan_end, 0, shared_mem_size);
    }
  } else {
    if (use_optimized) {
      printf("Using optimized global memory kernel\n");
      dedispersion_kernel_optimized<T><<<grids, threads>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, down_ndata, time_downsample,
          nchans, chan_start, chan_end, 0);
    } else {
      printf("Using legacy global memory kernel\n");
      dedispersion_kernel<T><<<grids, threads>>>(
          d_output, d_binned_input, d_delay_table, dm_steps, down_ndata, time_downsample,
          nchans, chan_start, chan_end, 0);
    }
  }

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  printf("Dedispersion completed in %lld ms\n", duration.count());

  // [SECOND FUNCTION - dedisperse_spec] Optimized memory copy - single large transfer to maximize PCIe bandwidth
  auto copy_start = std::chrono::high_resolution_clock::now();
  
  // Create aligned array directly for final storage
  auto dm_array_typed = std::shared_ptr<dedispersion_output_t<T>[]>(
      new (std::align_val_t{4096}) dedispersion_output_t<T>[dm_steps * down_ndata](),
      [](dedispersion_output_t<T> *p) { operator delete[](p, std::align_val_t{4096}); });
  
  // Single large transfer to maximize PCIe bandwidth utilization
  const size_t total_bytes = dm_steps * down_ndata * sizeof(dedispersion_output_t<T>);
  printf("Using single large memory copy for %zu MB to maximize PCIe bandwidth\n", total_bytes / (1024 * 1024));
  
  CHECK_CUDA(cudaMemcpy(dm_array_typed.get(), d_output,
                        total_bytes, cudaMemcpyDeviceToHost));
  
  auto copy_end = std::chrono::high_resolution_clock::now();
  auto copy_duration = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - copy_start);
  printf("Memory copy completed in %lld ms (%.2f GB/s)\n", 
         copy_duration.count(), 
         (total_bytes / 1024.0 / 1024.0 / 1024.0) / (copy_duration.count() / 1000.0));

  // Clean up GPU resources
  if (time_downsample > 1) {
    CHECK_CUDA(cudaFree(d_binned_input));
  } else {
    CHECK_CUDA(cudaFree(d_input)); // d_binned_input == d_input when no binning
  }
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_delay_table));
  CHECK_CUDA(cudaFree(d_freq_table));

  // Create typed dedispersion data structure - no additional copy needed
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

  printf("Full dedispersion completed. Now applying type-safe preprocessing with slicing...\n");
  Header updated_header = header;
  updated_header.tsamp = (time_downsample > 1) ? header.tsamp * time_downsample : header.tsamp;
  return preprocess_typed_dedisperseddata_with_slicing<T>(typed_result, updated_header, 1, t_sample);

}

template dedisperseddata_uint8
dedispered_fil_cuda<uint8_t>(Filterbank &fil, float dm_low, float dm_high,
                             float freq_start, float freq_end, float dm_step,
                             int ref_freq, int time_downsample, float t_sample, 
                             int target_id, std::string mask_file);

template dedisperseddata_uint8
dedispered_fil_cuda<uint16_t>(Filterbank &fil, float dm_low, float dm_high,
                              float freq_start, float freq_end, float dm_step,
                              int ref_freq, int time_downsample,
                              float t_sample, int target_id, std::string mask_file);

template dedisperseddata_uint8
dedispered_fil_cuda<uint32_t>(Filterbank &fil, float dm_low, float dm_high,
                              float freq_start, float freq_end, float dm_step,
                              int ref_freq, int time_downsample,
                              float t_sample, int target_id, std::string mask_file);

template dedisperseddata_uint8
dedisperse_spec<uint8_t>(uint8_t *data, Header header, float dm_low,
                         float dm_high, float freq_start, float freq_end,
                         float dm_step, int ref_freq, int time_downsample,
                         float t_sample, int target_id, std::string mask_file);

template dedisperseddata_uint8
dedisperse_spec<uint16_t>(uint16_t *data, Header header, float dm_low,
                          float dm_high, float freq_start, float freq_end,
                          float dm_step, int ref_freq, int time_downsample,
                          float t_sample, int target_id, std::string mask_file);
                          
template dedisperseddata_uint8
dedisperse_spec<uint32_t>(uint32_t *data, Header header, float dm_low,
                          float dm_high, float freq_start, float freq_end,
                          float dm_step, int ref_freq, int time_downsample,
                          float t_sample, int target_id, std::string mask_file);

} // namespace gpucal
