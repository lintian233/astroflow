#pragma once

#ifndef _CPUCAL_H
#define _CPUCAL_H

#include "data.h"
#include "filterbank.h"
#include "marcoutils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <vector>

using namespace std;

namespace cpucal {

#define REF_FREQ_END 1
#define REF_FREQ_START 0

/**
 * @brief 对滤波器数据去色散并生成时间样本
 *
 * 该函数对输入的滤波器数据（Filterbank）进行去色散处理，并生成一系列时间样本dm_times。
 * 去色散处理是通过计算不同色散量（DM）下的信号延迟，并对信号进行累加来实现的。
 * 函数支持多线程并行处理不同的DM步长。
 *
 * @tparam T 数据类型模板参数，通常为float或double
 * @param fil 输入的滤波器数据，包含频率表、时间采样间隔等信息
 * @param dm_low 色散量的下限
 * @param dm_high 色散量的上限
 * @param freq_start 频率起始(MHz)
 * @param freq_end 频率截至(MHz)
 * @param dm_step 色散量的步长，默认为1
 * @param ref_freq 参考频率，默认为通道最高频率（REF_FREQ_END）
 *  WARN: 暂时只能为REF_FREQ_END
 *
 * @param time_downsample 时间下采样因子，默认为64即每个64个fil.tsamp采一个
 * @param t_sample 时间样本的长度，默认为0.5秒
 *
 * @return DedisperedData<T>
 * 返回一个去色散后的数据类型
 *
 * @throws std::invalid_argument 如果输入参数不合法（如time_downsample <
 * 1，或dm_low > dm_high等）
 *
 * @note: 比omp实现稍快一些
 *
 * @details
 * 1.
 * 函数首先检查输入参数的合法性，包括time_downsample、dm_low、dm_high和dm_step。
 * 2.
 * 计算每个时间片的长度（samples_per_tsample）和总的时间片数量（total_slices）。
 * 3. 对于每个时间片，计算下采样后的时间点数（down_ndata）。
 * 4. 对于每个DM步长，计算信号在不同频率通道中的色散延迟，并对信号进行累加。
 * 5. 使用多线程并行处理不同的DM步长，以提高计算效率。
 * 6. 最后，将每个时间片的去色散结果存储在dm_times数组中，并返回该数组。
 *
 * @details 原理：
 * 色散是指不同频率的电磁波在介质中传播速度不同，导致信号在时间上产生延迟的现象。
 * 在射电天文中，色散效应会导致脉冲星信号在不同频率通道中产生时间延迟。
 * 该函数通过计算不同频率通道中的色散延迟，并对信号进行累加，从而消除色散效应。
 * 具体来说，函数通过以下步骤实现去色散：
 * 1. 计算每个频率通道中的色散延迟（delay）。
 * 2. 根据延迟计算目标时间点的索引（target_idx）。
 * 3. 对目标时间点的信号进行累加，得到去色散后的信号。
 */
template <typename T>
dedisperseddata
dedispered_fil_omp(Filterbank &fil, float dm_low, float dm_high,
                   float freq_start, float freq_end, float dm_step = 1,
                   int ref_freq = REF_FREQ_END, int time_downsample = 64,
                   float t_sample = 0.5) {

  // Set maximum number of OpenMP threads

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
  PRINT_VAR(chan_start);
  PRINT_VAR(chan_end);
  if (chan_start >= fil.nchans || chan_end >= fil.nchans ||
      chan_start >= chan_end)
    throw std::invalid_argument("Invalid chan parameters");
  if (time_downsample < 1)
    throw std::invalid_argument("time_downsample must be >= 1");
  if (dm_low > dm_high || dm_low < 0 || dm_step <= 0)
    throw std::invalid_argument("Invalid DM parameters");
  if (t_sample > fil.ndata * fil.tsamp)
    throw std::invalid_argument("t_sample exceeds total observation time");

  T *data = static_cast<T *>(fil.data);
  const size_t nchans = fil.nchans;
  const size_t dm_steps = static_cast<size_t>((dm_high - dm_low) / dm_step) + 1;

  const float ref_freq_value = ref_freq ? fil.frequency_table[chan_end]
                                        : fil.frequency_table[chan_start];

  dedisperseddata result;
  // 预计算所有DM值和通道的延迟样本数
  std::vector<std::vector<int>> delay_table(dm_steps);
#pragma omp parallel for
  for (size_t step = 0; step < dm_steps; ++step) {
    const float dm = dm_low + step * dm_step;
    std::vector<int> delays(chan_end - chan_start);
    for (size_t ch = chan_start; ch < chan_end; ++ch) {
      const float freq = fil.frequency_table[ch];
      const float delay =
          4148.808f * dm *
          (1.0f / (freq * freq) - 1.0f / (ref_freq_value * ref_freq_value));
      delays[ch - chan_start] = static_cast<int>(std::round(delay / fil.tsamp));
    }
    delay_table[step] = std::move(delays);
  }

  const size_t samples_per_tsample =
      static_cast<size_t>(std::round(t_sample / fil.tsamp));
  const size_t total_slices =
      (fil.ndata + samples_per_tsample - 1) / samples_per_tsample;
  std::vector<std::shared_ptr<uint64_t[]>> dm_times;

  // 主循环
  for (size_t slice_idx = 0; slice_idx < total_slices; ++slice_idx) {
    const size_t start = slice_idx * samples_per_tsample;
    const size_t end =
        std::min(start + samples_per_tsample, static_cast<size_t>(fil.ndata));
    const size_t slice_duration = end - start;
    PRINT_VAR(start);
    const size_t down_ndata =
        (slice_duration + time_downsample - 1) / time_downsample;
    if (slice_idx == 0) {
      result.downtsample_ndata = down_ndata;
      result.shape = {static_cast<size_t>(dm_steps),
                      static_cast<size_t>(down_ndata)};
      PRINT_VAR(down_ndata);
    }
    auto dm_array = std::shared_ptr<uint64_t[]>(
        new (std::align_val_t{4096}) uint64_t[dm_steps * down_ndata](),
        [](uint64_t *p) { operator delete[](p, std::align_val_t{4096}); });

// 并行处理DM步
#pragma omp parallel for schedule(dynamic)
    for (size_t step = 0; step < dm_steps; ++step) {
      const auto &delays = delay_table[step];
      const size_t step_offset = step * down_ndata;

      // 处理时间维度
      for (size_t ti = 0; ti < down_ndata; ++ti) {
        const size_t base_idx = start + ti * time_downsample;
        uint64_t sum = 0;

// SIMD优化循环
#pragma omp simd reduction(+ : sum)
        for (size_t ch = 0; ch < chan_end - chan_start; ++ch) {
          size_t target_idx = base_idx + delays[ch];
          if (target_idx < fil.ndata) {
            sum += data[target_idx * nchans + (ch + chan_start)];
          }
        }
        dm_array[step_offset + ti] = sum;
      }
    }

    dm_times.emplace_back(std::move(dm_array));
  }

  result.dm_times = std::move(dm_times);
  result.dm_low = dm_low;
  result.dm_high = dm_high;
  result.dm_step = dm_step;
  result.tsample = t_sample;
  result.filname = fil.filename;
  result.dm_ndata = dm_steps;

  return result;
}

template <typename T>
Spectrum<T> dedispered_fil_with_dm(Filterbank *fil, float tstart, float tend,
                                   float dm, float freq_start, float freq_end) {

  omp_set_num_threads(32);

  size_t t_start_idx = static_cast<size_t>(tstart / fil->tsamp);
  size_t t_end_idx = static_cast<size_t>(tend / fil->tsamp);
  size_t t_len = t_end_idx - t_start_idx;

  float fil_freq_min = fil->frequency_table[0];
  float fil_freq_max = fil->frequency_table[fil->nchans - 1];

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
                          (fil_freq_max - fil_freq_min) * (fil->nchans - 1));
  size_t chan_end =
      static_cast<size_t>((freq_end - fil_freq_min) /
                          (fil_freq_max - fil_freq_min) * (fil->nchans - 1));

  chan_start = std::max(static_cast<size_t>(0), chan_start);
  chan_end = std::min(static_cast<size_t>(fil->nchans - 1), chan_end);

  Spectrum<T> result;
  result.nbits = fil->nbits;
  result.ntimes = t_len;
  result.tstart = tstart;
  result.tend = tend;
  result.dm = dm;
  T *origin_data = static_cast<T *>(fil->data);

  result.nchans = chan_end - chan_start;
  result.freq_start = fil->frequency_table[chan_start];
  result.freq_end = fil->frequency_table[chan_end];
  result.data = std::shared_ptr<T[]>(new T[t_len * (result.nchans)](),
                                     [](T *p) { delete[] p; });

  int *dm_delays_table = new int[result.nchans];

#pragma omp parallel for schedule(dynamic)
  for (size_t ch = chan_start; ch < chan_end; ++ch) {
    const float freq = fil->frequency_table[ch];
    const float delay =
        4148.808f * dm *
        (1.0f / (freq * freq) - 1.0f / (fil->frequency_table[chan_end] *
                                        fil->frequency_table[chan_end]));

    dm_delays_table[ch - chan_start] =
        static_cast<int>(std::round(delay / fil->tsamp));
  }

#pragma omp parallel for schedule(dynamic)
  for (size_t ti = 0; ti < t_len; ++ti) {
#pragma omp simd
    for (size_t ch = chan_start; ch < chan_end; ++ch) {
      size_t target_idx = t_start_idx + ti + dm_delays_table[ch - chan_start];
      if (target_idx < fil->ndata) {
        result.data[ti * result.nchans + ch - chan_start] =
            origin_data[target_idx * fil->nchans + ch];
      } else {
        result.data[ti * result.nchans + ch - chan_start] = 0;
      }
    }
  }
  delete[] dm_delays_table;

  return result;
}

template <typename T>
Spectrum<T> dedisperse_spec_with_dm(T *spec, Header header, float dm,
                                    float tstart, float tend, float freq_start,
                                    float freq_end) {
  omp_set_num_threads(32);
  PRINT_VAR(header.tsamp);

  size_t t_start_idx = static_cast<size_t>(tstart / header.tsamp);
  size_t t_end_idx = static_cast<size_t>(tend / header.tsamp);
  size_t t_len = t_end_idx - t_start_idx;
  if (t_len <= 0) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg), "Invalid time range [%.3f-%.3f s]",
             tstart, tend);
    throw std::invalid_argument(error_msg);
  }
  PRINT_VAR(t_len);
  PRINT_VAR(t_start_idx);
  PRINT_VAR(t_end_idx);

  float *frequency_table = new float[header.nchans];
  for (size_t i = 0; i < header.nchans; i++) {
    frequency_table[i] = header.fch1 + i * header.foff;
  }

  float fil_freq_min = frequency_table[0];
  float fil_freq_max = frequency_table[header.nchans - 1];
  PRINT_VAR(fil_freq_min);
  PRINT_VAR(fil_freq_max);

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
                          (fil_freq_max - fil_freq_min) * (header.nchans - 1));
  size_t chan_end =
      static_cast<size_t>((freq_end - fil_freq_min) /
                          (fil_freq_max - fil_freq_min) * (header.nchans - 1));

  chan_start = std::max(static_cast<size_t>(0), chan_start);
  chan_end = std::min(static_cast<size_t>(header.nchans - 1), chan_end);

  Spectrum<T> result;
  result.nbits = header.nbits;
  result.ntimes = t_len;
  result.tstart = tstart;
  result.tend = tend;
  result.dm = dm;

  result.nchans = chan_end - chan_start;
  result.freq_start = frequency_table[chan_start];
  result.freq_end = frequency_table[chan_end];
  result.data = std::shared_ptr<T[]>(new T[t_len * (result.nchans)](),
                                     [](T *p) { delete[] p; });

  int *dm_delays_table = new int[result.nchans];

#pragma omp parallel for schedule(dynamic)
  for (size_t ch = chan_start; ch < chan_end; ++ch) {
    const float freq = frequency_table[ch];
    const float delay =
        4148.808f * dm *
        (1.0f / (freq * freq) -
         1.0f / (frequency_table[chan_end] * frequency_table[chan_end]));

    dm_delays_table[ch - chan_start] =
        static_cast<int>(std::round(delay / header.tsamp));
  }

#pragma omp parallel for schedule(dynamic)
  for (size_t ti = 0; ti < t_len; ++ti) {
#pragma omp simd
    for (size_t ch = chan_start; ch < chan_end; ++ch) {
      size_t target_idx = t_start_idx + ti + dm_delays_table[ch - chan_start];
      if (target_idx < header.ndata) {
        result.data[ti * result.nchans + ch - chan_start] =
            spec[target_idx * header.nchans + ch];
      } else {
        result.data[ti * result.nchans + ch - chan_start] = 0;
      }
    }
  }
  delete[] dm_delays_table;
  delete[] frequency_table;

  return result;
}
} // namespace cpucal
#endif //_CPUCAL_H
