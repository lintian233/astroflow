/**
 * Project astroflow
 */

#ifndef _GPUCAL_H
#define _GPUCAL_H

#include "data.h"
#include "filterbank.h"

namespace gpucal {

#define REF_FREQ_END 1
#define REF_FREQ_START 0

template <typename T>
dedisperseddata_uint8
dedispered_fil_cuda(Filterbank &fil, float dm_low, float dm_high,
                    float freq_start, float freq_end, float dm_step = 1,
                    int ref_freq = REF_FREQ_END, int time_downsample = 64,
                    float t_sample = 0.5, std::string mask_file = "mask.txt",
                    bool use_shared_memory = true);

template <typename T>
dedisperseddata_uint8 dedisperse_spec(T *data, Header header, float dm_low,
                                float dm_high, float freq_start, float freq_end,
                                float dm_step, int ref_freq = REF_FREQ_END,
                                int time_downsample = 64, float t_sample = 0.5,
                                std::string mask_file = "mask.txt",
                                bool use_shared_memory = true);

} // namespace gpucal
#endif //_GPUCAL_H
