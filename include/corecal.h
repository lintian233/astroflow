#pragma once

#ifndef _CORECAL_H
#define _CORECAL_H

#include "cpucal.hpp"
#include "data.h"
#include "gpucal.h"
#include <memory>

using deddata =
    std::variant<dedisperseddata<uint8_t>, dedisperseddata<uint16_t>,
                 dedisperseddata<uint32_t>>;

deddata dedisperse_data(Filterbank &fil, float dm_low, float dm_high,
                        float freq_start, float freq_end, float dm_step,
                        int time_downsample, float t_sample);

#endif //_CORECAL_H
