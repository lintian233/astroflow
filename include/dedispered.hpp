/**
 * dedispered.hpp
 *
 * Created on: Jan 30, 2025
 *
 * Author: (xd)[https://github.com/lintian233]
 *
 */

#pragma once

#include "filterbank.h"
#include "marcoutils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <future>
#include <memory>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace dedispered {

#define REF_FREQ_TOP 1
#define REF_FREQ_END 0

/** NOTE: -- doc generate by deepseek-v3: 2025-01-30
 *
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
 * @param dm_step 色散量的步长，默认为1
 * @param ref_freq 参考频率，默认为通道最高频率（REF_FREQ_TOP）
 *  WARN: 暂时只能为REF_FREQ_TOP
 *
 * @param time_downsample 时间下采样因子，默认为64即每个64个fil.tsamp采一个
 * @param t_sample 时间样本的长度，默认为0.5秒
 *
 * @return std::vector<std::shared_ptr<T[]>>:
 * 返回一个指针数组，每个指针指向一个去色散后的时间样本数组
 * std::vector<std::shared_ptr<T[]>> dm_times:每个时间片的dm_time图
 *
 * @throws std::invalid_argument 如果输入参数不合法（如time_downsample <
 * 1，或dm_low > dm_high等）
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
std::vector<std::shared_ptr<T[]>>
dedispered_fil_tsample(Filterbank &fil, float dm_low, float dm_high,
                       float dm_step = 1, int ref_freq = REF_FREQ_TOP,
                       int time_downsample = 64, float t_sample = 0.5) {

  size_t chan_start = 750;
  size_t chan_end = 1250;

  // 参数检查
  if (time_downsample < 1)
    throw std::invalid_argument("time_downsample must be >= 1");
  if (dm_low > dm_high || dm_low < 0 || dm_step <= 0) {
    throw std::invalid_argument("Invalid DM parameters");
  }
  if (t_sample > fil.ndata * fil.tsamp) {
    std::ostringstream oss;
    oss << "t_sample must less than total_tsample: " << t_sample
        << " total_tsample: " << fil.ndata * fil.tsamp;
    throw std::invalid_argument(oss.str());
  }

  T *data = static_cast<T *>(fil.data);
  const size_t nchans = fil.nchans;
  const float ref_freq_value = ref_freq ? fil.frequency_table[chan_end]
                                        : fil.frequency_table[chan_start];

  const int samples_per_tsample =
      static_cast<int>(std::round(t_sample / fil.tsamp));

  const size_t total_slices =
      (fil.ndata + samples_per_tsample - 1) / samples_per_tsample - 1;
  std::vector<std::shared_ptr<T[]>> dm_times;

  for (size_t slice_idx = 0; slice_idx < total_slices; ++slice_idx) {
    const size_t start = slice_idx * samples_per_tsample;
    const size_t end =
        std::min(start + samples_per_tsample, static_cast<size_t>(fil.ndata));

    PRINT_VAR(start);
    PRINT_VAR(end);
    const size_t slice_duration = end - start;
    const size_t down_ndata =
        (slice_duration + time_downsample - 1) / time_downsample;

    const int dm_steps = static_cast<int>((dm_high - dm_low) / dm_step) + 1;

    std::shared_ptr<T[]> dm_array(new T[dm_steps * down_ndata](),
                                  std::default_delete<T[]>());

    std::vector<std::future<void>> futures;

    auto process_dm = [&dm_array, data, &fil, start, time_downsample,
                       down_ndata, nchans, dm_steps, dm_low, dm_step,
                       chan_start, chan_end, ref_freq_value](int step) {
      const float dm = dm_low + step * dm_step;
      for (size_t ti = 0; ti < down_ndata; ++ti) {
        T sum = 0;
        const size_t base_idx = start + ti * time_downsample;

        for (size_t ch = chan_start; ch < chan_end; ++ch) {
          const float freq = fil.frequency_table[ch];
          const float delay =
              4.148808e3f * dm *
              (1.0f / (freq * freq) - 1.0f / (ref_freq_value * ref_freq_value));
          const int delay_samples = static_cast<int>(delay / fil.tsamp);

          const int target_idx = base_idx + delay_samples;

          if (target_idx >= 0 && static_cast<size_t>(target_idx) < fil.ndata) {
            sum += data[target_idx * nchans + ch];
          }
        }
        dm_array[step * down_ndata + ti] = sum;
      }
    };

    for (int step = 0; step < dm_steps; ++step) {
      futures.push_back(std::async(std::launch::async, process_dm, step));
    }
    for (auto &f : futures)
      f.get();

    dm_times.push_back(dm_array);
  }

  return dm_times;
}

/**
 * NOTE: dedispered_fil_tsample 的 openmp 版本
 *
 * doc 可以看dedispered_fil_tsample
 */
template <typename T>
std::vector<std::shared_ptr<T[]>>
dedispered_fil_tsample_omp(Filterbank &fil, float dm_low, float dm_high,
                           float dm_step = 1, int ref_freq = REF_FREQ_TOP,
                           int time_downsample = 64, float t_sample = 0.5) {
  int chan_start = 0;
  int chan_end = 1250;
  // 参数检查
  if (chan_start < 0 || chan_end >= fil.nchans)
    throw std::invalid_argument("Invalid chan parameters");
  if (time_downsample < 1)
    throw std::invalid_argument("time_downsample must be >= 1");
  if (dm_low > dm_high || dm_low < 0 || dm_step <= 0)
    throw std::invalid_argument("Invalid DM parameters");
  if (t_sample > fil.ndata * fil.tsamp)
    throw std::invalid_argument("t_sample exceeds total observation time");

  T *data = static_cast<T *>(fil.data);
  const size_t nchans = fil.nchans;
  const int dm_steps = static_cast<int>((dm_high - dm_low) / dm_step) + 1;

  const float ref_freq_value = ref_freq ? fil.frequency_table[chan_end]
                                        : fil.frequency_table[chan_start];

  // 预计算所有DM值和通道的延迟样本数
  std::vector<std::vector<int>> delay_table(dm_steps);
#pragma omp parallel for
  for (int step = 0; step < dm_steps; ++step) {
    const float dm = dm_low + step * dm_step;
    std::vector<int> delays(chan_end - chan_start);
    for (int ch = chan_start; ch < chan_end; ++ch) {
      const float freq = fil.frequency_table[ch];
      const float delay =
          4148.808f * dm *
          (1.0f / (freq * freq) - 1.0f / (ref_freq_value * ref_freq_value));
      delays[ch - chan_start] = static_cast<int>(std::round(delay / fil.tsamp));
    }
    delay_table[step] = std::move(delays);
  }

  const int samples_per_tsample =
      static_cast<int>(std::round(t_sample / fil.tsamp));
  const size_t total_slices =
      (fil.ndata + samples_per_tsample - 1) / samples_per_tsample;
  std::vector<std::shared_ptr<T[]>> dm_times;

  // 主循环
  for (size_t slice_idx = 0; slice_idx < total_slices; ++slice_idx) {
    const size_t start = slice_idx * samples_per_tsample;
    const size_t end =
        std::min(start + samples_per_tsample, static_cast<size_t>(fil.ndata));
    const size_t slice_duration = end - start;
    PRINT_VAR(start);
    const size_t down_ndata =
        (slice_duration + time_downsample - 1) / time_downsample;
    if (slice_idx == 0)
      PRINT_VAR(down_ndata);
    auto dm_array = std::shared_ptr<T[]>(
        new (std::align_val_t{4096}) T[dm_steps * down_ndata](),
        [](T *p) { operator delete[](p, std::align_val_t{4096}); });

// 并行处理DM步
#pragma omp parallel for schedule(dynamic)
    for (int step = 0; step < dm_steps; ++step) {
      const auto &delays = delay_table[step];
      const size_t step_offset = step * down_ndata;

      // 处理时间维度
      for (size_t ti = 0; ti < down_ndata; ++ti) {
        const size_t base_idx = start + ti * time_downsample;
        T sum = 0;

// SIMD优化循环
#pragma omp simd reduction(+ : sum)
        for (int ch = 0; ch < chan_end - chan_start; ++ch) {
          const int target_idx = base_idx + delays[ch];
          if (target_idx >= 0 && static_cast<size_t>(target_idx) < fil.ndata) {
            sum += data[target_idx * nchans + (ch + chan_start)];
          }
        }
        dm_array[step_offset + ti] = sum;
      }
    }

    dm_times.emplace_back(std::move(dm_array));
  }

  return dm_times;
}

} // namespace dedispered
