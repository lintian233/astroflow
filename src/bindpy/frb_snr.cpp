#include "pyapi.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

double compute_mean(const std::vector<double> &values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

double compute_std(const std::vector<double> &values, double mean) {
  if (values.empty()) {
    return 0.0;
  }
  double acc = 0.0;
  for (double v : values) {
    double diff = v - mean;
    acc += diff * diff;
  }
  return std::sqrt(acc / static_cast<double>(values.size()));
}

double compute_median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const size_t n = values.size();
  const size_t mid = n / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  double median = values[mid];
  if (n % 2 == 0) {
    double lower = *std::max_element(values.begin(), values.begin() + mid);
    median = 0.5 * (lower + median);
  }
  return median;
}

std::pair<double, double> robust_mean_std(const std::vector<double> &data,
                                          double sigma, int max_iter) {
  std::vector<double> finite_data;
  finite_data.reserve(data.size());
  for (double v : data) {
    if (std::isfinite(v)) {
      finite_data.push_back(v);
    }
  }

  if (finite_data.empty()) {
    return {0.0, 1.0};
  }

  std::vector<double> clipped = finite_data;
  for (int iter = 0; iter < max_iter; ++iter) {
    double med = compute_median(clipped);
    std::vector<double> abs_dev;
    abs_dev.reserve(clipped.size());
    for (double v : clipped) {
      abs_dev.push_back(std::fabs(v - med));
    }
    double mad = compute_median(abs_dev);
    double std = mad > 0.0 ? 1.4826 * mad : compute_std(clipped, compute_mean(clipped));
    if (std <= 0.0) {
      break;
    }
    std::vector<double> next_clipped;
    next_clipped.reserve(clipped.size());
    double limit = sigma * std;
    for (double v : clipped) {
      if (std::fabs(v - med) <= limit) {
        next_clipped.push_back(v);
      }
    }
    if (next_clipped.size() == clipped.size()) {
      break;
    }
    size_t min_size =
        std::max<size_t>(10, static_cast<size_t>(0.2 * clipped.size()));
    if (next_clipped.size() < min_size) {
      break;
    }
    clipped.swap(next_clipped);
  }

  double mean = clipped.empty() ? compute_mean(finite_data) : compute_mean(clipped);
  double std = clipped.empty() ? compute_std(finite_data, mean)
                               : compute_std(clipped, mean);
  if (std <= 0.0) {
    std = compute_std(finite_data, compute_mean(finite_data));
  }
  if (std <= 0.0) {
    std = 1.0;
  }
  return {mean, std};
}

std::vector<int> build_widths(int max_width) {
  if (max_width <= 1) {
    return {1};
  }
  if (max_width <= 16) {
    std::vector<int> widths;
    widths.reserve(max_width);
    for (int i = 1; i <= max_width; ++i) {
      widths.push_back(i);
    }
    return widths;
  }

  std::vector<int> widths;
  widths.reserve(32);
  for (int i = 1; i <= 16; ++i) {
    widths.push_back(i);
  }
  const int num = 32;
  double log_min = std::log10(17.0);
  double log_max = std::log10(static_cast<double>(max_width));
  std::vector<int> logspace;
  logspace.reserve(num);
  for (int i = 0; i < num; ++i) {
    double t = num == 1 ? 0.0 : static_cast<double>(i) / (num - 1);
    double value = std::pow(10.0, log_min + t * (log_max - log_min));
    int rounded = static_cast<int>(std::llround(value));
    if (rounded >= 17 && rounded <= max_width) {
      logspace.push_back(rounded);
    }
  }
  std::sort(logspace.begin(), logspace.end());
  logspace.erase(std::unique(logspace.begin(), logspace.end()), logspace.end());
  widths.insert(widths.end(), logspace.begin(), logspace.end());
  return widths;
}

} // namespace

py::tuple calculate_frb_snr_cpp(py::array spec, py::object noise_range,
                                double threshold_sigma, py::object toa_sample_idx,
                                py::object fitting_window_samples, py::object tsamp,
                                py::object target_time_us) {
  auto spec_array =
      py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(spec);
  if (!spec_array) {
    throw std::runtime_error("spec must be a 2D numpy array");
  }

  auto info = spec_array.request();
  if (info.ndim != 2) {
    throw std::runtime_error("spec must be 2D (time x frequency)");
  }

  const int n_time_orig = static_cast<int>(info.shape[0]);
  const int n_freq = static_cast<int>(info.shape[1]);
  if (n_time_orig <= 0 || n_freq <= 0) {
    py::dict fit_quality;
    fit_quality["fit_converged"] = false;
    return py::make_tuple(-1.0, 0, 0, py::make_tuple(0.0, 1.0, fit_quality));
  }

  const float *data_ptr = static_cast<float *>(info.ptr);
  std::vector<double> time_series_raw;
  time_series_raw.resize(n_time_orig);
  for (int t = 0; t < n_time_orig; ++t) {
    double sum = 0.0;
    const float *row = data_ptr + static_cast<size_t>(t) * n_freq;
    for (int f = 0; f < n_freq; ++f) {
      sum += row[f];
    }
    time_series_raw[t] = sum;
  }

  int downsample = 1;
  bool have_tsamp = !tsamp.is_none();
  bool have_target = !target_time_us.is_none();
  if (have_tsamp && have_target) {
    double tsamp_val = tsamp.cast<double>();
    double target_us = target_time_us.cast<double>();
    if (tsamp_val > 0.0) {
      double target_sec = target_us * 1e-6;
      downsample = static_cast<int>(std::llround(target_sec / tsamp_val));
    }
    if (downsample < 1) {
      downsample = 1;
    }
  }

  if (downsample > 1) {
    int trimmed = (n_time_orig / downsample) * downsample;
    if (trimmed >= downsample) {
      int n_groups = trimmed / downsample;
      std::vector<double> downsampled;
      downsampled.resize(n_groups);
      for (int g = 0; g < n_groups; ++g) {
        double sum = 0.0;
        int base = g * downsample;
        for (int i = 0; i < downsample; ++i) {
          sum += time_series_raw[base + i];
        }
        downsampled[g] = sum;
      }
      time_series_raw.swap(downsampled);
    } else {
      downsample = 1;
    }
  }

  const int n_time = static_cast<int>(time_series_raw.size());
  if (n_time <= 0) {
    py::dict fit_quality;
    fit_quality["fit_converged"] = false;
    return py::make_tuple(-1.0, 0, 0, py::make_tuple(0.0, 1.0, fit_quality));
  }

  int max_width = 0;
  if (fitting_window_samples.is_none()) {
    max_width = std::max(8, std::min((n_time * downsample) / 4, 512));
  } else {
    max_width = std::max(4, std::min(fitting_window_samples.cast<int>(),
                                     (n_time * downsample) / 2));
  }
  max_width = std::max(1, std::min(max_width, n_time * downsample));
  int max_width_ds = static_cast<int>(
      std::ceil(static_cast<double>(max_width) / downsample));
  max_width_ds = std::min(max_width_ds, n_time);

  int search_start = 0;
  int search_end = n_time;
  bool have_toa = !toa_sample_idx.is_none();
  if (have_toa) {
    int search_radius = std::max(max_width_ds,
                                 static_cast<int>(std::ceil(100.0 / downsample)));
    int toa_idx = toa_sample_idx.cast<int>() / downsample;
    search_start = std::max(0, toa_idx - search_radius);
    search_end = std::min(n_time, toa_idx + search_radius);
  }

  std::vector<double> noise_data;
  if (!noise_range.is_none() && py::len(noise_range) > 0) {
    py::iterable ranges = noise_range;
    for (py::handle item : ranges) {
      auto tuple = py::reinterpret_borrow<py::tuple>(item);
      if (tuple.size() != 2) {
        continue;
      }
      int start = tuple[0].cast<int>();
      int end = tuple[1].cast<int>();
      int ds_start = std::max(0, start / downsample);
      int ds_end = std::min(n_time,
                            static_cast<int>(std::ceil(static_cast<double>(end) /
                                                       downsample)));
      if (ds_end > ds_start) {
        noise_data.insert(noise_data.end(), time_series_raw.begin() + ds_start,
                          time_series_raw.begin() + ds_end);
      }
    }
    if (noise_data.empty()) {
      noise_data = time_series_raw;
    }
  } else {
    if (have_toa && (search_start > 0 || search_end < n_time)) {
      if (search_start > 0) {
        noise_data.insert(noise_data.end(), time_series_raw.begin(),
                          time_series_raw.begin() + search_start);
      }
      if (search_end < n_time) {
        noise_data.insert(noise_data.end(), time_series_raw.begin() + search_end,
                          time_series_raw.end());
      }
    }
    if (noise_data.size() < 10) {
      noise_data = time_series_raw;
    }
  }

  auto noise_stats = robust_mean_std(noise_data, threshold_sigma, 3);
  double noise_mean = noise_stats.first;
  double noise_std = noise_stats.second;

  std::vector<int> widths = build_widths(max_width_ds);
  std::vector<double> cumsum;
  cumsum.resize(static_cast<size_t>(n_time) + 1);
  cumsum[0] = 0.0;
  for (int i = 0; i < n_time; ++i) {
    cumsum[static_cast<size_t>(i) + 1] = cumsum[static_cast<size_t>(i)] +
                                        time_series_raw[static_cast<size_t>(i)];
  }

  double best_snr = -std::numeric_limits<double>::infinity();
  int best_width = 1;
  int best_center = 0;
  double best_sum = std::numeric_limits<double>::quiet_NaN();

  if (!time_series_raw.empty()) {
    auto max_it = std::max_element(time_series_raw.begin(), time_series_raw.end());
    best_center = static_cast<int>(std::distance(time_series_raw.begin(), max_it));
  }

  for (int width : widths) {
    if (width >= n_time) {
      break;
    }
    int window_count = n_time + 1 - width;
    int center_start = width / 2;
    bool found = false;
    double local_best_snr = -std::numeric_limits<double>::infinity();
    int local_best_center = best_center;
    double local_best_sum = 0.0;

    for (int i = 0; i < window_count; ++i) {
      int center = center_start + i;
      if (have_toa && (center < search_start || center >= search_end)) {
        continue;
      }
      double sum = cumsum[static_cast<size_t>(i) + width] - cumsum[static_cast<size_t>(i)];
      double snr = -std::numeric_limits<double>::infinity();
      if (noise_std > 0.0) {
        snr = (sum - noise_mean * width) /
              (noise_std * std::sqrt(static_cast<double>(width)));
      }
      if (!found || snr > local_best_snr) {
        found = true;
        local_best_snr = snr;
        local_best_center = center;
        local_best_sum = sum;
      }
    }

    if (found && local_best_snr > best_snr) {
      best_snr = local_best_snr;
      best_width = width;
      best_center = local_best_center;
      best_sum = local_best_sum;
    }
  }

  if (!std::isfinite(best_sum)) {
    auto fallback_stats = robust_mean_std(time_series_raw, threshold_sigma, 3);
    noise_mean = fallback_stats.first;
    noise_std = fallback_stats.second;
    int peak_idx = 0;
    if (!time_series_raw.empty()) {
      auto max_it = std::max_element(time_series_raw.begin(), time_series_raw.end());
      peak_idx = static_cast<int>(std::distance(time_series_raw.begin(), max_it));
    }
    double snr = noise_std > 0.0 ? (time_series_raw[peak_idx] - noise_mean) / noise_std
                                 : -1.0;
    py::dict fit_quality;
    fit_quality["fit_converged"] = false;
    fit_quality["method"] = "fallback";
    int peak_idx_orig = std::min(n_time_orig - 1,
                                 peak_idx * downsample + downsample / 2);
    return py::make_tuple(snr, std::max(1, downsample), peak_idx_orig,
                          py::make_tuple(noise_mean, noise_std, fit_quality));
  }

  int refine_left = std::max(0, best_center - 2 * best_width);
  int refine_right = std::min(n_time, best_center + 2 * best_width + 1);
  std::vector<double> refine_data;
  refine_data.reserve(n_time);
  for (int i = 0; i < n_time; ++i) {
    if (i < refine_left || i >= refine_right) {
      refine_data.push_back(time_series_raw[static_cast<size_t>(i)]);
    }
  }
  if (refine_data.size() >=
      std::max<size_t>(10, static_cast<size_t>(0.1 * n_time))) {
    auto refined = robust_mean_std(refine_data, threshold_sigma, 3);
    noise_mean = refined.first;
    noise_std = refined.second;
  }

  double snr = noise_std > 0.0
                   ? (best_sum - noise_mean * best_width) /
                         (noise_std * std::sqrt(static_cast<double>(best_width)))
                   : -1.0;

  py::dict fit_quality;
  fit_quality["fit_converged"] = true;
  fit_quality["method"] = "boxcar";
  fit_quality["width_samples"] = best_width * downsample;
  fit_quality["search_start"] = search_start;
  fit_quality["search_end"] = search_end;
  fit_quality["downsample"] = downsample;

  int best_center_orig = std::min(n_time_orig - 1,
                                  best_center * downsample + downsample / 2);
  int best_width_orig = std::max(1, best_width * downsample);

  return py::make_tuple(snr, best_width_orig, best_center_orig,
                        py::make_tuple(noise_mean, noise_std, fit_quality));
}
