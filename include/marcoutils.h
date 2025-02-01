#pragma once

#ifndef MARCOUTILS_H_
#define MARCOUTILS_H_

#include "matplotlibcpp.h"
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace MarcUtils {

// Function to print the first 10 elements of a vector
template <typename T> void print_vector(const std::vector<T> &vec) {
  std::cout << "[ ";
  for (size_t i = 0; i < vec.size() && i < 10; ++i) {
    std::cout << vec[i] << " ";
  }
  std::cout << "]" << std::endl;
}

// Function to format a string with given arguments
template <typename... Args>
std::string format(const std::string &format_str, Args &&...args) {
  std::ostringstream oss;
  size_t arg_index = 0;
  size_t pos = 0;
  const size_t num_args = sizeof...(args);

  std::initializer_list<int>{
      (oss << (pos == 0
                   ? ""
                   : format_str.substr(pos, format_str.find("{}", pos) - pos))
           << (arg_index < num_args ? std::forward<Args>(args) : ""),
       pos = format_str.find("{}", pos) + 2, ++arg_index)...};

  oss << format_str.substr(pos);
  return oss.str();
}

// Function to display an image using matplotlib
template <typename T> void imshow(const T *ptr, std::vector<size_t> shape) {
  namespace plt = matplotlibcpp;
  std::map<std::string, std::string> kwargs = {
      {"cmap", "viridis"}, {"vmin", "0"}, {"vmax", "200"}, {"aspect", "auto"}};

  plt::imshow(ptr, shape[0], shape[1], 1, kwargs);
  plt::tight_layout();
  plt::show();
  // plt::save("dm_time.png", 300);
}

} // namespace MarcUtils

// Macro for formatted printing
#define PRINT_FORMAT(FORMAT_STR, ...)                                          \
  std::cout << MarcUtils::format(FORMAT_STR, __VA_ARGS__) << std::endl;

// Debug macros
#define DEBUG
#ifdef DEBUG
#define PRINT_VEC(VEC) MarcUtils::print_vector(VEC)
#define IMSHOW(PTR, SHAPE) MarcUtils::imshow(PTR, SHAPE)
#define PRINT_VAR(VAR) (std::cout << #VAR << ": " << (VAR) << std::endl)
#else
#define PRINT_VEC(VEC)                                                         \
  do {                                                                         \
  } while (0)
#define IMSHOW(PTR, SHAPE)                                                     \
  do {                                                                         \
  } while (0)
#define PRINT_VAR(VAR)                                                         \
  do {                                                                         \
  } while (0)
#endif // DEBUG

#endif // MARCOUTILS_H_
