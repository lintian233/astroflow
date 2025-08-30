/**
 * Project astroflow
 */
#pragma once
#ifndef _GPUCAL_H
#define _GPUCAL_H

#include "data.h"
#include "rfi.h"
#include "filterbank.h"

namespace gpucal {

#define REF_FREQ_END 1
#define REF_FREQ_START 0

template <typename T>
dedisperseddata_uint8
dedispered_fil_cuda(Filterbank &fil, float dm_low, float dm_high,
                    float freq_start, float freq_end, float dm_step,
                    int ref_freq, int time_downsample, 
                    float t_sample, int target_id, std::string mask_file, rficonfig rficfg);

template <typename T>
dedisperseddata_uint8 dedisperse_spec(T *data, Header header, float dm_low,
                                float dm_high, float freq_start, float freq_end,
                                float dm_step, int ref_freq,
                                int time_downsample, float t_sample, int target_id,
                                std::string mask_file, rficonfig rficfg);


template <typename T>
Spectrum<T> dedisperse_spec_with_dm(T *spec, Header header, float dm,
                                    float tstart, float tend,
                                    float freq_start, float freq_end,
                                    std::string maskfile, rficonfig rficfg);

template <typename T>
Spectrum<T> dedispered_fil_with_dm(Filterbank *fil, float tstart, float tend,
                                   float dm, float freq_start, float freq_end,
                                   std::string maskfile, rficonfig rficfg);
                                   
} // namespace gpucal
#endif //_GPUCAL_H
