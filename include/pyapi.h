#pragma once

#ifndef _PYAPI_H
#define _PYAPI_H

#include "cpucal.hpp"
#include "data.h"
#include "gpucal.h"
#include <cstdint>
#include <memory>
#include <string>

namespace py = pybind11;

#define CPU_TARGET 0
#define GPU_TARGET 1

dedisperseddata_uint8 dedispered_fil(std::string filename, float dm_low,
                               float dm_high, float freq_start, float freq_end,
                               float dm_step, int time_downsample,
                               float t_sample, int target, int target_id,
                               std::string mask_file);

template <typename T>
dedisperseddata_uint8 
dedisperse_spec_py(py::array_t<T> data, Header header, float dm_low,
                   float dm_high, float freq_start, float freq_end,
                   float dm_step, int time_downsample, float t_sample, int target_id,
                   std::string mask_file);

template <typename T>
Spectrum<T> dedisperse_spec_with_dm_py(py::array_t<T> data, Header header,
                                       float dm, float tstart, float tend,
                                       float freq_start, float freq_end, 
                                       std::string maskfile);

template <typename T>
void bind_dedispersed_data(py::module &m, const char *class_name);

void bind_dedispersed_data_uint8(py::module &m, const char *class_name);

void bind_filterbank(py::module &m);

template <typename T> void bind_spectrum(py::module &m, const char *class_name);

void bind_header(py::module &m);

#endif
