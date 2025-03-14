#include "pyapi.h"
#include "corecal.h"
#include "cpucal.hpp"
#include "data.h"
#include "filterbank.h"
#include <cstdint>

/* template <typename T> void bind_dedispersed_data(py::module &m) {
  using Data = dedisperseddata<T>;
  py::class_<Data>(m, "DedispersedData")
      .def(py::init<>())
      .def_property_readonly("dm_times",
                             [](const Data &data) {
                               std::vector<py::array_t<T>> py_arrays;
                               for (const auto &ptr : data.dm_times) {
                                 // 假设每个数组的大小是downtsample_ndata
                                 size_t size = data.downtsample_ndata;
                                 // 直接映射C++内存到NumPy数组
                                 auto arr = py::array_t<T>({size}, ptr.get());
                                 arr.writeable(
                                     false); // 设置为只读，防止Python修改数据
                                 py_arrays.push_back(arr);
                               }
                               return py::cast(py_arrays);
                             })
      .def_readonly("shape", &Data::shape)
      .def_readonly("dm_ndata", &Data::dm_ndata)
      .def_readonly("downtsample_ndata", &Data::downtsample_ndata)
      .def_readonly("dm_low", &Data::dm_low)
      .def_readonly("dm_high", &Data::dm_high)
      .def_readonly("dm_step", &Data::dm_step)
      .def_readonly("tsample", &Data::tsample)
      .def_readonly("filname", &Data::filname);
} */

/* template <typename T>
dedisperseddata<T>
dedispered_fil(std::string filename, float dm_low, float dm_high,
               float freq_start, float freq_end, float dm_step = 1,
               int time_downsample = 64, float t_sample = 0.5, int njobs = 64) {
  Filterbank fil(filename);
  fil.info();
  omp_set_num_threads(njobs);
  return cpucal::dedispered_fil_omp<T>(fil, dm_low, dm_high, freq_start,
                                       freq_end, dm_step, REF_FREQ_END,
                                       time_downsample, t_sample);
}

// 显式模板实例化
template dedisperseddata<uint8_t> dedispered_fil<uint8_t>(std::string, float,
                                                          float, float, float,
                                                          float, int, float,
                                                          int);

template dedisperseddata<uint16_t> dedispered_fil<uint16_t>(std::string, float,
                                                            float, float, float,
                                                            float, int, float,
                                                            int); */
