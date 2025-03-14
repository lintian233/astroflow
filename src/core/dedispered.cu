#include "gpucal.h"
#include <cuda_runtime.h>

/* namespace gpucal {

__constant__ float c_freq_table[4096];  // 最大支持4096通道
__constant__ int c_delays[4096 * 1024]; // 最大支持1024 DM步×4096通道

template <typename T>
__global__ void
dedisperse_kernel(const T *__restrict__ data, T *__restrict__ output,
                  int nchans, size_t ndata, int chan_start, int chan_count,
                  int dm_steps, int down_ndata, int time_downsample) {
  const int dm_step = blockIdx.x;
  const int time_idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (dm_step >= dm_steps || time_idx >= down_ndata)
    return;

  const size_t base_idx = blockIdx.z * ndata + time_idx * time_downsample;
  T sum = 0;

  const int *delays = c_delays + dm_step * chan_count;

  for (int ch = 0; ch < chan_count; ++ch) {
    const int target_idx = base_idx + delays[ch];
    if (target_idx >= 0 && target_idx < ndata) {
      sum += data[target_idx * nchans + (ch + chan_start)];
    }
  }

  output[dm_step * down_ndata + time_idx] = sum;
}

template <typename T>
dedisperseddata<T> dedispered_fil_cuda(Filterbank &fil, float dm_low,
                                       float dm_high, float freq_start,
                                       float freq_end, float dm_step,
                                       int ref_freq, int time_downsample,
                                       float t_sample, int block_size) {
  T *d_data;
  cudaMalloc(&d_data, fil.ndata * fil.nchans * sizeof(T));
  cudaMemcpy(d_data, fil.data, fil.ndata * fil.nchans * sizeof(T),
             cudaMemcpyHostToDevice);

  // 拷贝频率表到常量内存
  cudaMemcpyToSymbol(c_freq_table, fil.frequency_table,
                     fil.nchans * sizeof(float));

  const int dm_steps = static_cast<int>((dm_high - dm_low) / dm_step) + 1;
  std::vector<int> h_delays(dm_steps * (chan_end - chan_start));

  // 在Host端生成延迟表
  const float ref_freq_value = ref_freq ? fil.frequency_table[chan_end]
                                        : fil.frequency_table[chan_start];
  for (int step = 0; step < dm_steps; ++step) {
    const float dm = dm_low + step * dm_step;
    for (int ch = chan_start; ch < chan_end; ++ch) {
      const float freq = fil.frequency_table[ch];
      const float delay =
          4148.808f * dm *
          (1.0f / (freq * freq) - 1.0f / (ref_freq_value * ref_freq_value));
      h_delays[step * (chan_end - chan_start) + (ch - chan_start)] =
          static_cast<int>(std::round(delay / fil.tsamp));
    }
  }

  // 拷贝延迟表到常量内存
  cudaMemcpyToSymbol(c_delays, h_delays.data(), h_delays.size() * sizeof(int));

  // 分片处理逻辑
  const int samples_per_tsample = static_cast<int>(t_sample / fil.tsamp);
  const size_t total_slices =
      (fil.ndata + samples_per_tsample - 1) / samples_per_tsample;

  dedisperseddata<T> result;
  result.dm_low = dm_low;
  result.dm_high = dm_high;
  result.dm_step = dm_step;
  result.tsamp = t_sample;
  result.filname = fil.filename;
  result.dm_ndata = dm_steps;

  for (size_t slice_idx = 0; slice_idx < total_slices; ++slice_idx) {
    const size_t start = slice_idx * samples_per_tsample;
    const size_t end =
        std::min(start + samples_per_tsample, static_cast<size_t>(fil.ndata));
    const size_t slice_duration = end - start;
    const size_t down_ndata =
        (slice_duration + time_downsample - 1) / time_downsample;

    // 分配设备内存
    T *d_output;
    cudaMalloc(&d_output, dm_steps * down_ndata * sizeof(T));
    cudaMemset(d_output, 0, dm_steps * down_ndata * sizeof(T));

    // 配置内核参数
    dim3 grid(dm_steps, (down_ndata + block_size - 1) / block_size, 1);
    dedisperse_kernel<T><<<grid, block_size>>>(
        d_data + start * fil.nchans, // 当前分片起始地址
        d_output, fil.nchans, slice_duration, chan_start, chan_end - chan_start,
        dm_steps, down_ndata, time_downsample);

    // 拷贝结果回Host
    auto slice_result = std::shared_ptr<T[]>(new T[dm_steps * down_ndata]);
    cudaMemcpy(slice_result.get(), d_output, dm_steps * down_ndata * sizeof(T),
               cudaMemcpyDeviceToHost);
    result.dm_times.emplace_back(std::move(slice_result));

    cudaFree(d_output);
  }

  cudaFree(d_data);
  return result;
}


// 显式实例化模板
template dedisperseddata<uint8_t>
dedispered_fil_cuda<uint8_t>(Filterbank &, float, float, float, float, float,
                             int, int, float, int);
template dedisperseddata<uint16_t>
dedispered_fil_cuda<uint16_t>(Filterbank &, float, float, float, float, float,
                              int, int, float, int);
} */
