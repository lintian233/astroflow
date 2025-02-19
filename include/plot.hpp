#pragma once

#include <algorithm>
#ifndef PLOT_H_
#define PLOT_H_

#include "dedispered.hpp"
#include "marcoutils.h"
#include "matplotlibcpp.h"
#include <filesystem>
#include <sstream>
#include <string>

template <typename T>
void plot_dm_time_imp(const T *ptr, std::vector<size_t> shape,
                      std::string title, bool save = true) {
  namespace plt = matplotlibcpp;
  T max_val = *std::max_element(ptr, ptr + shape[0] * shape[1]);
  T min_val = *std::min_element(ptr, ptr + shape[0] * shape[1]);

  std::map<std::string, std::string> kwargs = {
      {"cmap", "viridis"},
      {"vmin", std::to_string(min_val)},
      {"vmax", std::to_string(max_val)},
      {"aspect", "auto"}};

  plt::imshow(ptr, shape[0], shape[1], 1, kwargs);
  plt::ylabel("DM");
  plt::xlabel("TIME");
  plt::title(title);
  plt::tight_layout();
  if (save) {
    string savename = title + "dm_time.png";
    plt::save(savename, 300);
  } else {
    plt::show();
  }
}

template <typename T>
void plot_dedispered_data_imp(dedispered::DedispersedData<T> &ddata,
                              bool save = true) {
  PRINT_VAR(ddata.dm_times.size());
  for (int i = 0; i < ddata.dm_times.size(); i++) {
    std::stringstream title;
    std::filesystem::path path = ddata.filname;
    title << path.stem().string() << "_" << i << "_";
    plot_dm_time_imp<T>(ddata.dm_times[i].get(), ddata.shape, title.str(),
                        save);
  }
}

#define plot_dedispered_data(dedata, save)                                     \
  plot_dedispered_data_imp(dedata, save)

#endif // PLOT_H_
