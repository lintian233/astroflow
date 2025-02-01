#pragma once

#ifndef ASTROFUNC_H_
#define ASTROFUNC_H_

#include "dedispered.hpp"
#include "filterbank.h"
#include "marcoutils.h"
#include "plot.hpp"

void single_pulsar_search(Filterbank &fil, float dm_low, float dm_high,
                          float freq_start, float freq_end, float dm_step,
                          int time_downsample, float t_sample);

#endif // ASTROFUNC_H_
