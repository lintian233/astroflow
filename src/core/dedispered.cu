#include "gpucal.h"
#include "marcoutils.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector_types.h>
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
template <typename T>
__global__ void
dedispersion_kernel(uint32_t *output, T *input, int *delay_table, int dm_steps,
                    int time_downsample, int ndata, int nchans, int chan_start,
                    int chan_end, int start, int down_ndata) {
  int dmidx = blockIdx.y;
  int down_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dmidx >= dm_steps)
    return;
  if (down_idx >= down_ndata)
    return;

  int base_idx = down_idx * time_downsample + start;
  uint32_t sum = 0;
  for (int chan = chan_start; chan < chan_end; ++chan) {
    int target_idx =
        base_idx +
        delay_table[dmidx * (chan_end - chan_start + 1) + chan - chan_start];
    /* if (down_idx == 4000 && dmidx == 200) {
      printf(
          "chan %d, target_idx %d, delay %d\n", chan, target_idx,
          delay_table[dmidx * (chan_end - chan_start + 1) + chan - chan_start]);
    } */
    if (target_idx > 0 && static_cast<size_t>(target_idx) < ndata) {
      sum += input[chan + target_idx * nchans];
    }
  }
  /* if (down_idx == 4000) {
    printf("sum %d\n", sum);
  } */
  output[dmidx * down_ndata + down_idx] = sum;
}

__global__ void
pre_calculate_dedispersion_kernel(int *delay_table, float dm_low, float dm_high,
                                  float dm_step, int chan_start, int chan_end,
                                  double *freq_table, float ref_freq_value,
                                  double tsamp) {

  int dmidx = blockDim.x * blockIdx.x + threadIdx.x;
  float dm = dm_low + (blockDim.x * blockIdx.x + threadIdx.x) * (dm_step);
  if (dm > dm_high)
    return;
  int chan = blockDim.y * blockIdx.y + threadIdx.y + chan_start;
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
dedisperseddata dedispered_fil_cuda(Filterbank &fil, float dm_low,
                                    float dm_high, float freq_start,
                                    float freq_end, float dm_step, int ref_freq,
                                    int time_downsample, float t_sample) {

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
  int chan_start =
      static_cast<int>((freq_start - fil_freq_min) /
                       (fil_freq_max - fil_freq_min) * (fil.nchans - 1));
  int chan_end =
      static_cast<int>((freq_end - fil_freq_min) /
                       (fil_freq_max - fil_freq_min) * (fil.nchans - 1));

  chan_start = std::max(0, chan_start);
  chan_end = std::min(fil.nchans - 1, chan_end);

  if (chan_start < 0 || chan_end >= fil.nchans) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "Invalid channel range [%d-%d] for %d channels", chan_start,
             chan_end, fil.nchans);
    throw std::invalid_argument(error_msg);
  }
  if (time_downsample < 1) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "Invalid time_downsample value %d, "
             "must be greater than 1",
             time_downsample);
    throw std::invalid_argument(error_msg);
  }
  if (dm_low > dm_high || dm_low < 0 || dm_step <= 0) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "Invalid DM range [%.3f-%.3f] with step %.3f", dm_low, dm_high,
             dm_step);
  }
  if (t_sample > fil.ndata * fil.tsamp) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "Invalid t_sample value %.3f, must be less than %.3f", t_sample,
             fil.ndata * fil.tsamp);
  }

  const size_t nchans = fil.nchans;
  const size_t dm_steps = static_cast<int>((dm_high - dm_low) / dm_step) + 1;

  const float ref_freq_value = ref_freq ? fil.frequency_table[chan_end]
                                        : fil.frequency_table[chan_start];

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
  // check the delay table
  int *delay_table = new int[dm_steps * (chan_end - chan_start + 1)];
  CHECK_CUDA(cudaMemcpy(delay_table, d_delay_table,
                        dm_steps * (chan_end - chan_start + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost));
  PRINT_ARR(delay_table, dm_steps * (chan_end - chan_start + 1));

  const int samples_per_tsample =
      static_cast<int>(std::round(t_sample / fil.tsamp));
  const size_t total_slices =
      (fil.ndata + samples_per_tsample - 1) / samples_per_tsample;

  const int down_ndata_t =
      (samples_per_tsample + time_downsample - 1) / time_downsample;

  T *d_input;
  uint32_t *d_output;
  T *data = static_cast<T *>(fil.data);
  // print class T
  PRINT_VAR(sizeof(T));
  CHECK_CUDA(cudaMalloc(&d_input, fil.ndata * nchans * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_output, dm_steps * down_ndata_t * sizeof(uint32_t)));
  CHECK_CUDA(
      cudaMemset(d_output, 0, dm_steps * down_ndata_t * sizeof(uint32_t)));

  CHECK_CUDA(cudaMemcpy(d_input, data, fil.ndata * nchans * sizeof(T),
                        cudaMemcpyHostToDevice));

  dedisperseddata result;

  std::vector<std::shared_ptr<uint32_t[]>> dm_times;

  for (size_t slice_idx = 0; slice_idx < total_slices; ++slice_idx) {
    const size_t start = slice_idx * samples_per_tsample;
    const size_t end =
        std::min(start + samples_per_tsample, static_cast<size_t>(fil.ndata));
    const size_t slice_duration = end - start;
    const size_t down_ndata =
        (slice_duration + time_downsample - 1) / time_downsample;
    PRINT_VAR(start);
    if (slice_idx == 0) {
      result.downtsample_ndata = down_ndata;
      result.shape = {dm_steps, down_ndata};
      PRINT_VAR(down_ndata);
      PRINT_VAR(result.shape[0]);
      PRINT_VAR(result.shape[1]);
    }
    CHECK_CUDA(
        cudaMemset(d_output, 0, dm_steps * down_ndata_t * sizeof(uint32_t)));

    int THREADS_PER_BLOCK = 256;
    dim3 threads(THREADS_PER_BLOCK);
    dim3 grids(down_ndata + threads.x - 1 / threads.x, dm_steps);

    dedispersion_kernel<T><<<grids, threads>>>(
        d_output, d_input, d_delay_table, dm_steps, time_downsample, fil.ndata,
        nchans, chan_start, chan_end, start, down_ndata);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    auto dm_array = std::shared_ptr<uint32_t[]>(
        new (std::align_val_t{4096}) uint32_t[dm_steps * down_ndata_t](),
        [](uint32_t *p) { operator delete[](p, std::align_val_t{4096}); });

    CHECK_CUDA(cudaMemcpy(dm_array.get(), d_output,
                          dm_steps * down_ndata * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    dm_times.emplace_back(std::move(dm_array));
  }
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));

  result.dm_times = std::move(dm_times);
  result.dm_low = dm_low;
  result.dm_high = dm_high;
  result.dm_step = dm_step;
  result.tsample = t_sample;
  result.filname = fil.filename;
  result.dm_ndata = dm_steps;

  PRINT_VAR(result.filname);
  PRINT_VAR(result.dm_times.size());
  PRINT_ARR(result.dm_times[0].get(), 10);
  return result;
}

template dedisperseddata
dedispered_fil_cuda<uint8_t>(Filterbank &fil, float dm_low, float dm_high,
                             float freq_start, float freq_end, float dm_step,
                             int ref_freq, int time_downsample, float t_sample);

template dedisperseddata
dedispered_fil_cuda<uint16_t>(Filterbank &fil, float dm_low, float dm_high,
                              float freq_start, float freq_end, float dm_step,
                              int ref_freq, int time_downsample,
                              float t_sample);

template dedisperseddata
dedispered_fil_cuda<uint32_t>(Filterbank &fil, float dm_low, float dm_high,
                              float freq_start, float freq_end, float dm_step,
                              int ref_freq, int time_downsample,
                              float t_sample);

} // namespace gpucal
