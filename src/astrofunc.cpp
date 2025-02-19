#include "astrofunc.h"

using DedispersedData = std::variant<dedispered::DedispersedData<uint8_t>,
                                     dedispered::DedispersedData<uint16_t>,
                                     dedispered::DedispersedData<uint32_t>>;

DedispersedData dedisperse_data(Filterbank &fil, float dm_low, float dm_high,
                                float freq_start, float freq_end, float dm_step,
                                int time_downsample, float t_sample) {
  switch (fil.nbits) {
  case 8:
    return dedispered::dedispered_fil_tsample_omp<uint8_t>(
        fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
        time_downsample, t_sample);
  case 16:
    return dedispered::dedispered_fil_tsample_omp<uint16_t>(
        fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
        time_downsample, t_sample);
  case 32:
    return dedispered::dedispered_fil_tsample_omp<uint32_t>(
        fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
        time_downsample, t_sample);
  default:
    throw std::runtime_error(
        "Unsupported data format. Supported formats are 8, 16, and 32 bits.");
  }
}

void single_pulsar_search(Filterbank &fil, float dm_low, float dm_high,
                          float freq_start, float freq_end, float dm_step,
                          int time_downsample, float t_sample) {
  // 调用去色散函数
  DedispersedData dedata =
      dedisperse_data(fil, dm_low, dm_high, freq_start, freq_end, dm_step,
                      time_downsample, t_sample);

  auto de_vistor = [](auto &&arg) {
    using T = std::decay_t<decltype(arg)>; // Get the actual type
    plot_dedispered_data(arg, false);
    std::cout << "DM range: [" << arg.dm_low << ", " << arg.dm_high << "]\n";
    std::cout << "DM step: " << arg.dm_step << "\n";
    std::cout << "File name: " << arg.filname << "\n";
  };
  // 使用 std::visit 访问 dedata
  std::visit(de_vistor, dedata);
}
