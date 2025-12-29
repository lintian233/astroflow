#pragma once

#ifndef _CPUCAL_H
#define _CPUCAL_H

#include "data.h"
#include "filterbank.h"
#include "marcoutils.h"
#include "rfimarker_cpu.h"
#include "iqrm.hpp"
#include "iqrmcuda.h"
#include "rfi.h"
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
dedisperseddata_uint8
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

  return preprocess_dedisperseddata(result, 512);
}

template <typename T>
Spectrum<T> dedispered_fil_with_dm(
    Filterbank* fil,
    float tstart, float tend,
    float dm,
    float freq_start, float freq_end,
    std::string maskfile,
    rficonfig rficfg)
{
    omp_set_num_threads(32);

    // ---- 基本时间索引 ----
    if (tend <= tstart) {
        throw std::invalid_argument("tend must be > tstart");
    }
    const size_t t_start_idx = static_cast<size_t>(tstart / fil->tsamp);
    const size_t t_end_idx   = static_cast<size_t>(tend   / fil->tsamp);
    size_t t_len_req         = (t_end_idx > t_start_idx) ? (t_end_idx - t_start_idx) : 0;
    if (t_len_req == 0 || t_start_idx >= fil->ndata) {
        throw std::invalid_argument("Invalid time window for this file.");
    }

    // ---- 频道范围（半开区间）并兼容升/降序 ----
    const float f0 = fil->frequency_table[0];
    const float fN = fil->frequency_table[fil->nchans - 1];
    const float fmin = std::min(f0, fN);
    const float fmax = std::max(f0, fN);
    if (freq_start < fmin || freq_end > fmax || freq_end <= freq_start) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Frequency range [%.3f, %.3f] MHz is out of file range [%.3f, %.3f] MHz",
                 freq_start, freq_end, fmin, fmax);
        throw std::invalid_argument(buf);
    }

    auto freq_to_index = [&](float f)->size_t {
        // 线性映射；对升/降序都成立
        double pos = (f - f0) / (double)(fN - f0);            // ∈[0,1]
        double raw = pos * (double)(fil->nchans - 1);
        if (raw < 0.0) raw = 0.0;
        if (raw > (double)(fil->nchans - 1)) raw = (double)(fil->nchans - 1);
        return static_cast<size_t>(std::floor(raw + 1e-9));   // 向下取整
    };

    size_t chan_start      = freq_to_index(freq_start);
    size_t chan_end_incl   = freq_to_index(freq_end);
    if (chan_end_incl < chan_start) std::swap(chan_start, chan_end_incl);
    size_t chan_end_excl   = std::min(chan_end_incl + 1, static_cast<size_t>(fil->nchans));
    const size_t sel_nch   = (chan_end_excl > chan_start) ? (chan_end_excl - chan_start) : 0;
    if (sel_nch == 0) {
        throw std::invalid_argument("Empty channel selection.");
    }

    // 频段内的最低/最高频（与升/降序无关）
    const float f_low  = std::min(fil->frequency_table[chan_start],
                                  fil->frequency_table[chan_end_excl - 1]);
    const float f_high = std::max(fil->frequency_table[chan_start],
                                  fil->frequency_table[chan_end_excl - 1]);

    // ---- 每通道延时表（参考频率取最高频 -> 延时最小为0）----
    std::unique_ptr<int[]> dm_delays(new int[sel_nch]);
    int delay_max_idx = 0;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t ch = (ptrdiff_t)chan_start; ch < (ptrdiff_t)chan_end_excl; ++ch) {
        const float fch   = fil->frequency_table[ch];
        const float delay = 4148.808f * dm * (1.0f/(fch*fch) - 1.0f/(f_high*f_high)); // 秒
        int d_idx = (int)std::lround(delay / fil->tsamp); // 采样点
        dm_delays[ch - chan_start] = d_idx;
    }
    // 求最大延时
    for (size_t i = 0; i < sel_nch; ++i) delay_max_idx = std::max(delay_max_idx, dm_delays[i]);

    // ---- 有效输出长度：必须保证访问 t_start_idx + ti + delay_max_idx < fil->ndata ----
    size_t t_len_eff_cap = (fil->ndata > t_start_idx)
                           ? (fil->ndata - t_start_idx)
                           : 0;
    size_t t_len_eff = 0;
    if (t_len_eff_cap > (size_t)delay_max_idx) {
        t_len_eff = std::min(t_len_req, t_len_eff_cap - (size_t)delay_max_idx);
    }
    if (t_len_eff == 0) {
        throw std::invalid_argument("Time window too short for this DM and band (no valid samples after dedispersion).");
    }

    // ---- RFI：只在“局部输入切片”上运行 ----
    // 扩展窗口以覆盖解色散所需的所有样本
    T* origin_data = static_cast<T*>(fil->data);
    size_t rfi_t_start_idx = (t_start_idx > (size_t)delay_max_idx) ? (t_start_idx - delay_max_idx) : 0;
    size_t rfi_offset_from_t_start = t_start_idx - rfi_t_start_idx;
    T* slice_ptr_for_rfi = origin_data + rfi_t_start_idx * fil->nchans;
    size_t slice_len_for_rfi = std::min(t_len_eff + rfi_offset_from_t_start + 2 * delay_max_idx, (size_t)fil->ndata - rfi_t_start_idx);

    RfiMarkerCPU<T> rfi_marker(maskfile);
    if (rficfg.use_iqrm) {
        auto win_masks = iqrm_cuda::rfi_iqrm_gpu_host<T>(
            slice_ptr_for_rfi,       // 指向扩展切片的起点
            chan_start, chan_end_excl,
            slice_len_for_rfi,       // 覆盖局部 + 最大延时
            fil->nchans,
            fil->tsamp, rficfg);
        rfi_marker.mask(slice_ptr_for_rfi, fil->nchans, slice_len_for_rfi, win_masks);
    }
    if (rficfg.use_mask) {
        // 静态掩膜同样只作用在扩展切片上
        rfi_marker.mark_rfi(slice_ptr_for_rfi, fil->nchans, slice_len_for_rfi);
    }

    // ---- 输出光谱 ----
    Spectrum<T> result;
    result.nbits       = fil->nbits;
    result.ntimes      = t_len_eff;
    result.tstart      = tstart;
    result.tend        = tstart + (float)t_len_eff * fil->tsamp;
    result.dm          = dm;
    result.nchans      = sel_nch;
    result.freq_start  = std::min(fil->frequency_table[chan_start],
                                  fil->frequency_table[chan_end_excl - 1]);
    result.freq_end    = std::max(fil->frequency_table[chan_start],
                                  fil->frequency_table[chan_end_excl - 1]);
    result.data        = std::shared_ptr<T[]>(new T[(size_t)result.ntimes * (size_t)result.nchans](),
                                              [](T* p){ delete[] p; });

    // ---- 解色散填充（只写有效长度 t_len_eff）----
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t ti = 0; ti < (ptrdiff_t)result.ntimes; ++ti) {
#pragma omp simd
        for (ptrdiff_t ch = (ptrdiff_t)chan_start; ch < (ptrdiff_t)chan_end_excl; ++ch) {
            const int d = dm_delays[ch - chan_start];
            const size_t src_idx = t_start_idx + (size_t)ti + (size_t)d;  // 相对于 origin_data 的绝对偏移
            result.data[(size_t)ti * (size_t)result.nchans + (size_t)(ch - chan_start)]
                = origin_data[src_idx * fil->nchans + (size_t)ch];
        }
    }

    return result;
}


template <typename T>
Spectrum<T> dedisperse_spec_with_dm(
    T* spec, Header header, float dm,
    float tstart, float tend,
    float freq_start, float freq_end,
    std::string maskfile, rficonfig rficfg)
{
    omp_set_num_threads(32);
    
    // ---- 时间窗口 ----
    if (tend <= tstart) {
        throw std::invalid_argument("tend must be > tstart");
    }
    size_t t_start_idx = static_cast<size_t>(tstart / header.tsamp);
    size_t t_end_idx   = static_cast<size_t>(tend   / header.tsamp);
    size_t t_len_req   = (t_end_idx > t_start_idx) ? (t_end_idx - t_start_idx) : 0;
    if (t_len_req == 0 || t_start_idx >= header.ndata) {
        throw std::invalid_argument("Invalid time window for this file.");
    }

    if (freq_start >= freq_end) {
        throw std::invalid_argument("freq_end must be > freq_start");
    }

    if (dm < 0.0f) {
        throw std::invalid_argument("dm must be >= 0");
    }

    if (header.foff < 0) {
        throw std::invalid_argument("frequency channels are in descending order, which is not supported yet.");
    }

    // ---- 构造频率表 ----
    std::vector<float> frequency_table(header.nchans);
    for (size_t i = 0; i < header.nchans; i++) {
        frequency_table[i] = header.fch1 + i * header.foff;
    }
    float f0   = frequency_table.front();
    float fN   = frequency_table.back();
    float fmin = std::min(f0, fN);
    float fmax = std::max(f0, fN);

    if (freq_start < fmin || freq_end > fmax || freq_end <= freq_start) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Frequency range [%.3f, %.3f] MHz out of file range [%.3f, %.3f] MHz",
                 freq_start, freq_end, fmin, fmax);
        throw std::invalid_argument(buf);
    }

    auto freq_to_index = [&](float f)->size_t {
        double pos = (f - f0) / (double)(fN - f0);
        double raw = pos * (double)(header.nchans - 1);
        if (raw < 0.0) raw = 0.0;
        if (raw > (double)(header.nchans - 1)) raw = (double)(header.nchans - 1);
        return static_cast<size_t>(std::floor(raw + 1e-9));
    };

    size_t chan_start    = freq_to_index(freq_start);
    size_t chan_end_incl = freq_to_index(freq_end);
    if (chan_end_incl < chan_start) std::swap(chan_start, chan_end_incl);
    size_t chan_end_excl = std::min(chan_end_incl + 1, static_cast<size_t>(header.nchans));

    size_t sel_nch = (chan_end_excl > chan_start) ? (chan_end_excl - chan_start) : 0;
    if (sel_nch == 0) {
        throw std::invalid_argument("Empty channel selection.");
    }

    // ---- 计算延时表 ----
    const float f_high = std::max(frequency_table[chan_start],
                                  frequency_table[chan_end_excl - 1]);
    std::unique_ptr<int[]> dm_delays(new int[sel_nch]);
    int delay_max_idx = 0;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t ch = (ptrdiff_t)chan_start; ch < (ptrdiff_t)chan_end_excl; ++ch) {
        float fch   = frequency_table[ch];
        float delay = 4148.808f * dm * (1.0f/(fch*fch) - 1.0f/(f_high*f_high));
        int d_idx   = (int)std::lround(delay / header.tsamp);
        dm_delays[ch - chan_start] = d_idx;
    }
    for (size_t i = 0; i < sel_nch; ++i) delay_max_idx = std::max(delay_max_idx, dm_delays[i]);

    // ---- 输出时间长度：必须保证访问有效 ----
    size_t t_len_cap = (header.ndata > t_start_idx)
                       ? (header.ndata - t_start_idx)
                       : 0;
    size_t t_len_eff = 0;
    if (t_len_cap > (size_t)delay_max_idx) {
        t_len_eff = std::min(t_len_req, t_len_cap - (size_t)delay_max_idx);
    }
    if (t_len_eff == 0) {
        throw std::invalid_argument("Time window too short for this DM and band.");
    }

    // ---- RFI（局部+最大延时）----
    size_t rfi_t_start_idx = (t_start_idx > (size_t)delay_max_idx) ? (t_start_idx - delay_max_idx) : 0;
    size_t rfi_offset_from_t_start = t_start_idx - rfi_t_start_idx;
    T* slice_ptr_for_rfi = spec + rfi_t_start_idx * header.nchans;
    size_t slice_len_for_rfi = std::min(t_len_eff + rfi_offset_from_t_start + 2 * delay_max_idx, header.ndata - rfi_t_start_idx);

    RfiMarkerCPU<T> rfi_marker(maskfile);
    if (rficfg.use_iqrm) {
        auto win_masks = iqrm_cuda::rfi_iqrm_gpu_host<T>(
            slice_ptr_for_rfi,
            chan_start, chan_end_excl,
            slice_len_for_rfi,
            header.nchans,
            header.tsamp, rficfg);
        rfi_marker.mask(slice_ptr_for_rfi, header.nchans, slice_len_for_rfi, win_masks);
    }
    if (rficfg.use_mask) {
        rfi_marker.mark_rfi(slice_ptr_for_rfi, header.nchans, slice_len_for_rfi);
    }

    // ---- 构造输出 ----
    Spectrum<T> result;
    result.nbits      = header.nbits;
    result.ntimes     = t_len_eff;
    result.tstart     = tstart;
    result.tend       = tstart + (float)t_len_eff * header.tsamp;
    result.dm         = dm;
    result.nchans     = sel_nch;
    result.freq_start = std::min(frequency_table[chan_start],
                                 frequency_table[chan_end_excl - 1]);
    result.freq_end   = std::max(frequency_table[chan_start],
                                 frequency_table[chan_end_excl - 1]);
    result.data       = std::shared_ptr<T[]>(new T[result.ntimes * result.nchans](),
                                             [](T* p){ delete[] p; });

    // ---- 解色散 ----
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t ti = 0; ti < (ptrdiff_t)result.ntimes; ++ti) {
#pragma omp simd
        for (ptrdiff_t ch = (ptrdiff_t)chan_start; ch < (ptrdiff_t)chan_end_excl; ++ch) {
            int d = dm_delays[ch - chan_start];
            size_t src_idx = t_start_idx + (size_t)ti + (size_t)d; // 相对于 spec
            result.data[(size_t)ti * result.nchans + (size_t)(ch - chan_start)]
                = spec[src_idx * header.nchans + (size_t)ch];
        }
    }

    return result;
}


} // namespace cpucal
#endif //_CPUCAL_H
