#include "corecal.h"

dedisperseddata dedisperse_data(Filterbank &fil, float dm_low, float dm_high,
                                float freq_start, float freq_end, float dm_step,
                                int time_downsample, float t_sample) {
  switch (fil.nbits) {
  case 8:
    return cpucal::dedispered_fil_omp<uint8_t>(fil, dm_low, dm_high, freq_start,
                                               freq_end, dm_step, REF_FREQ_END,
                                               time_downsample, t_sample);
  case 16:
    return cpucal::dedispered_fil_omp<uint16_t>(
        fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
        time_downsample, t_sample);
  case 32:
    return cpucal::dedispered_fil_omp<uint32_t>(
        fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
        time_downsample, t_sample);
  default:
    throw std::runtime_error(
        "Unsupported data format. Supported formats are 8, 16, and 32 bits.");
  }
}
