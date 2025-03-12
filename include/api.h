#pragma once

#include "cpucal.hpp"
#include "data.h"
#include <cstdint>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

template <typename T>
dedisperseddata<T>
dedispered_fil(std::string filename, float dm_low, float dm_high,
               float freq_start, float freq_end, float dm_step = 1,
               int time_downsample = 64, float t_sample = 0.5, int njobs = 64) {
  Filterbank fil(filename);
  fil.info();
  omp_set_num_threads(njobs);
  return cpucal::dedispered_fil_omp<T>(fil, dm_low, dm_high, freq_start,
                                       freq_end, dm_step, REF_FREQ_END,
                                       time_downsample, t_sample);
}

// 显式模板实例化
template dedisperseddata<uint8_t> dedispered_fil<uint8_t>(std::string, float,
                                                          float, float, float,
                                                          float, int, float,
                                                          int);

template dedisperseddata<uint16_t> dedispered_fil<uint16_t>(std::string, float,
                                                            float, float, float,
                                                            float, int, float,
                                                            int);
