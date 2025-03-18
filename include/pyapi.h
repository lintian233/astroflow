#pragma once

#include "cpucal.hpp"
#include "data.h"
#include "gpucal.h"
#include <cstdint>
#include <memory>
#include <string>

#define CPU_TARGET 0
#define GPU_TARGET 1

dedisperseddata dedispered_fil(std::string filename, float dm_low,
                               float dm_high, float freq_start, float freq_end,
                               float dm_step = 1, int time_downsample = 64,
                               float t_sample = 0.5, int target = 0) {
  Filterbank fil(filename);
  fil.info();
  switch (fil.nbits) {
  case 8: {
    if (target == GPU_TARGET) {
      return gpucal::dedispered_fil_cuda<uint8_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    } else if (target == CPU_TARGET) {
      return cpucal::dedispered_fil_omp<uint8_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    }
  }
  case 16: {
    if (target == GPU_TARGET) {
      return gpucal::dedispered_fil_cuda<uint16_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    } else if (target == CPU_TARGET) {
      return cpucal::dedispered_fil_omp<uint16_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    }
  }
  case 32: {
    if (target == GPU_TARGET) {
      return gpucal::dedispered_fil_cuda<uint32_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    } else if (target == CPU_TARGET) {
      return cpucal::dedispered_fil_omp<uint32_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    }
  }
  }
};
