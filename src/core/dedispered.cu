#include "data.h"
#include "gpucal.h"
#include "marcoutils.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector_types.h>
#include "rfimarker.h"

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

// Shared memory based dedispersion kernel implementing Algorithm 3
template <typename T>
__global__ void
dedispersion_shared_memory_kernel(uint64_t *output, T *input, int *delay_table,
                                  size_t dm_steps, int time_downsample, 
                                  size_t ndata, size_t nchans, 
                                  size_t chan_start, size_t chan_end,
                                  size_t start, size_t down_ndata,
                                  size_t shared_mem_size) {
  
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
  uint64_t Sloc = 0;
  
  // Initial time index
  size_t Tini = start + blockIdx.x * blockDim.x * time_downsample;
  
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
        // Calculate the time index with dedispersion delay
        size_t delay = delay_table[dmidx * Dch + chan - chan_start];
        size_t time_idx = Tini + threadIdx.x * time_downsample + delay;
        
        // Bounds checking
        if (time_idx < ndata) {
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

template <typename T>
__global__ void
dedispersion_kernel(uint64_t *output, T *input, int *delay_table,
                    size_t dm_steps, int time_downsample, size_t ndata,
                    size_t nchans, size_t chan_start, size_t chan_end,
                    size_t start, size_t down_ndata) {
  size_t dmidx = blockIdx.y;
  size_t down_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dmidx >= dm_steps)
    return;

    
  if (down_idx >= down_ndata)
    return;

  size_t base_idx = down_idx * time_downsample + start;
  uint64_t sum = 0;
  for (size_t chan = chan_start; chan < chan_end; ++chan) {
    size_t target_idx =
        base_idx +
        delay_table[dmidx * (chan_end - chan_start + 1) + chan - chan_start];
    if (target_idx > 0 && target_idx < ndata) {
      sum += input[chan + target_idx * nchans];
    } else {
      sum += 0;
    }
  }
  /* if (down_idx == 4000) {
    printf("sum %d\n", sum);
  } */
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
dedisperseddata_uint8 dedispered_fil_cuda(Filterbank &fil, float dm_low,
                                    float dm_high, float freq_start,
                                    float freq_end, float dm_step, int ref_freq,
                                    int time_downsample, float t_sample, 
                                    std::string mask_file, bool use_shared_memory) {

  // get all cuda devices
  int device_count;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    throw std::runtime_error("No CUDA devices found");
  }

  int device_id;
  if (device_count == 1) {
    device_id = 0;
  } else if (device_count == 4) {
    device_id = 2;
  } else {
    device_id = 0;
  }
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
  uint64_t *d_output;
  T *data = static_cast<T *>(fil.data);
  RfiMarker<T> rfi_marker(mask_file);
  rfi_marker.mark_rfi(data, fil.nchans, fil.ndata);

  CHECK_CUDA(cudaMalloc(&d_input, fil.ndata * nchans * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_output, dm_steps * down_ndata * sizeof(uint64_t)));
  CHECK_CUDA(cudaMemset(d_output, 0, dm_steps * down_ndata * sizeof(uint64_t)));

  CHECK_CUDA(cudaMemcpy(d_input, data, fil.ndata * nchans * sizeof(T),
                        cudaMemcpyHostToDevice));

  printf("Processing full data: DM steps = %zu, Time samples = %zu\n", dm_steps, down_ndata);

  int THREADS_PER_BLOCK = 256;
  dim3 threads(THREADS_PER_BLOCK);
  dim3 grids((down_ndata + threads.x - 1) / threads.x, dm_steps);

  if (use_shared_memory) {
    // Calculate shared memory size needed
    size_t max_shared_mem = device_prop.sharedMemPerBlock;
    size_t shared_mem_size = std::min(max_shared_mem / sizeof(T), 
                                     (chan_end - chan_start + 1) * THREADS_PER_BLOCK);
    
    // Ensure we don't exceed shared memory limits
    size_t actual_shared_mem = shared_mem_size * sizeof(T);
    
    printf("Using shared memory kernel with %zu bytes of shared memory\n", actual_shared_mem);
    
    dedispersion_shared_memory_kernel<T><<<grids, threads, actual_shared_mem>>>(
        d_output, d_input, d_delay_table, dm_steps, time_downsample, fil.ndata,
        nchans, chan_start, chan_end, 0, down_ndata, shared_mem_size);
  } else {
    printf("Using global memory kernel\n");
    dedispersion_kernel<T><<<grids, threads>>>(
        d_output, d_input, d_delay_table, dm_steps, time_downsample, fil.ndata,
        nchans, chan_start, chan_end, 0, down_ndata);
  }

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy back the full result
  auto dm_array = std::shared_ptr<uint64_t[]>(
      new (std::align_val_t{4096}) uint64_t[dm_steps * down_ndata](),
      [](uint64_t *p) { operator delete[](p, std::align_val_t{4096}); });

  CHECK_CUDA(cudaMemcpy(dm_array.get(), d_output,
                        dm_steps * down_ndata * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_delay_table));
  CHECK_CUDA(cudaFree(d_freq_table));

  // Create single large dedisperseddata with all time samples
  dedisperseddata result;
  result.dm_times.emplace_back(std::move(dm_array));
  result.dm_low = dm_low;
  result.dm_high = dm_high;
  result.dm_step = dm_step;
  result.tsample = t_sample;
  result.filname = fil.filename;
  result.dm_ndata = dm_steps;
  result.downtsample_ndata = down_ndata;
  result.shape = {dm_steps, down_ndata};

  printf("Full dedispersion completed. Now applying preprocessing with slicing...\n");
  // 对于Filterbank，需要构造Header结构
  Header temp_header;
  temp_header.tsamp = fil.tsamp;
  temp_header.filename = fil.filename;
  return preprocess_dedisperseddata_with_slicing(result, temp_header, time_downsample, t_sample);
}

template <typename T>
dedisperseddata_uint8 dedisperse_spec(T *data, Header header, float dm_low,
                                float dm_high, float freq_start, float freq_end,
                                float dm_step, int ref_freq,
                                int time_downsample, float t_sample, 
                                std::string mask_file, bool use_shared_memory) { 
  // get all cuda devices
  int device_count;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    throw std::runtime_error("No CUDA devices found");
  }

  int device_id;
  if (device_count == 1) {
    device_id = 0;
  } else if (device_count == 4) {
    device_id = 2;
  } else {
    device_id = 0;
  }
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

  RfiMarker<T> rfi_marker(mask_file);
  rfi_marker.mark_rfi(data, header.nchans, header.ndata);

  T *d_input;
  uint64_t *d_output;
  CHECK_CUDA(cudaMalloc(&d_input, header.ndata * nchans * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_output, dm_steps * down_ndata * sizeof(uint64_t)));
  CHECK_CUDA(cudaMemset(d_output, 0, dm_steps * down_ndata * sizeof(uint64_t)));

  CHECK_CUDA(cudaMemcpy(d_input, data, header.ndata * nchans * sizeof(T),
                        cudaMemcpyHostToDevice));

  printf("Processing full data: DM steps = %zu, Time samples = %zu\n", dm_steps, down_ndata);

  int THREADS_PER_BLOCK = 256;
  dim3 threads(THREADS_PER_BLOCK);
  dim3 grids((down_ndata + threads.x - 1) / threads.x, dm_steps);

  if (use_shared_memory) {
    // Calculate shared memory size needed
    size_t max_shared_mem = device_prop.sharedMemPerBlock;
    size_t shared_mem_size = std::min(max_shared_mem / sizeof(T), 
                                     (chan_end - chan_start + 1) * THREADS_PER_BLOCK);
    
    // Ensure we don't exceed shared memory limits
    size_t actual_shared_mem = shared_mem_size * sizeof(T);
    
    printf("Using shared memory kernel with %zu bytes of shared memory\n", actual_shared_mem);
    
    dedispersion_shared_memory_kernel<T><<<grids, threads, actual_shared_mem>>>(
        d_output, d_input, d_delay_table, dm_steps, time_downsample, header.ndata,
        nchans, chan_start, chan_end, 0, down_ndata, shared_mem_size);
  } else {
    printf("Using global memory kernel\n");
    dedispersion_kernel<T><<<grids, threads>>>(
        d_output, d_input, d_delay_table, dm_steps, time_downsample,
        header.ndata, nchans, chan_start, chan_end, 0, down_ndata);
  }

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy back the full result
  auto dm_array = std::shared_ptr<uint64_t[]>(
      new (std::align_val_t{4096}) uint64_t[dm_steps * down_ndata](),
      [](uint64_t *p) { operator delete[](p, std::align_val_t{4096}); });

  CHECK_CUDA(cudaMemcpy(dm_array.get(), d_output,
                        dm_steps * down_ndata * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));

  // Clean up GPU resources
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_delay_table));
  CHECK_CUDA(cudaFree(d_freq_table));

  // Create single large dedisperseddata with all time samples
  dedisperseddata result;
  result.dm_times.emplace_back(std::move(dm_array));
  result.dm_low = dm_low;
  result.dm_high = dm_high;
  result.dm_step = dm_step;
  result.tsample = t_sample;
  result.filname = header.filename;
  result.dm_ndata = dm_steps;
  result.downtsample_ndata = down_ndata;
  result.shape = {dm_steps, down_ndata};

  printf("Full dedispersion completed. Now applying preprocessing with slicing...\n");
  return preprocess_dedisperseddata_with_slicing(result, header, time_downsample, t_sample);

}

template dedisperseddata_uint8
dedispered_fil_cuda<uint8_t>(Filterbank &fil, float dm_low, float dm_high,
                             float freq_start, float freq_end, float dm_step,
                             int ref_freq, int time_downsample, float t_sample, 
                             std::string mask_file, bool use_shared_memory);

template dedisperseddata_uint8
dedispered_fil_cuda<uint16_t>(Filterbank &fil, float dm_low, float dm_high,
                              float freq_start, float freq_end, float dm_step,
                              int ref_freq, int time_downsample,
                              float t_sample, std::string mask_file, bool use_shared_memory);

template dedisperseddata_uint8
dedispered_fil_cuda<uint32_t>(Filterbank &fil, float dm_low, float dm_high,
                              float freq_start, float freq_end, float dm_step,
                              int ref_freq, int time_downsample,
                              float t_sample, std::string mask_file, bool use_shared_memory);

template dedisperseddata_uint8
dedisperse_spec<uint8_t>(uint8_t *data, Header header, float dm_low,
                         float dm_high, float freq_start, float freq_end,
                         float dm_step, int ref_freq, int time_downsample,
                         float t_sample, std::string mask_file, bool use_shared_memory);

template dedisperseddata_uint8
dedisperse_spec<uint16_t>(uint16_t *data, Header header, float dm_low,
                          float dm_high, float freq_start, float freq_end,
                          float dm_step, int ref_freq, int time_downsample,
                          float t_sample, std::string mask_file, bool use_shared_memory);
                          
template dedisperseddata_uint8
dedisperse_spec<uint32_t>(uint32_t *data, Header header, float dm_low,
                          float dm_high, float freq_start, float freq_end,
                          float dm_step, int ref_freq, int time_downsample,
                          float t_sample, std::string mask_file, bool use_shared_memory);

} // namespace gpucal
