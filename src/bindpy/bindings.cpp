#include "api.h"
#include "cpp_code.h"
#include <cstdint>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::shared_ptr<uint16_t[]> create_shared_data(size_t size) {
  return std::shared_ptr<uint16_t[]>(new uint16_t[size],
                                     [](uint16_t *p) { delete[] p; });
}

py::array_t<uint16_t> get_data() {
  const size_t length = 1000000;
  auto data_ptr = create_shared_data(length);

  for (size_t i = 0; i < length; ++i) {
    data_ptr[i] = i % 0xFFFF;
  }

  auto capsule =
      py::capsule(new std::shared_ptr<uint16_t[]>(data_ptr), [](void *p) {
        delete static_cast<std::shared_ptr<uint16_t[]> *>(p);
      });

  return py::array_t<uint16_t>({length},           // shape
                               {sizeof(uint16_t)}, // stride
                               data_ptr.get(),     // 数据指针
                               capsule);           // 所有权胶囊
}

void process_data(py::array_t<uint16_t> arr) {
  py::buffer_info buf = arr.request();
  uint16_t *data = static_cast<uint16_t *>(buf.ptr);
  size_t size = buf.size;

  for (size_t i = 0; i < size; ++i) {
    data[i] *= 2;
  }
}

template <typename T>
void bind_dedispersed_data(py::module &m, const char *class_name) {
  using Data = dedisperseddata<T>;
  py::class_<Data>(m, class_name)
      .def(py::init<>())
      .def_property_readonly("dm_times",
                             [](const Data &data) {
                               std::vector<py::array_t<T>> py_arrays;
                               for (const auto &ptr : data.dm_times) {
                                 int size = data.shape[0] * data.shape[1];
                                 // 直接映射C++内存到NumPy数组
                                 auto arr = py::array_t<T>({size}, ptr.get());
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
}

PYBIND11_MODULE(_astroflow_core, m) {

  py::class_<VectorAdder>(m, "VectorAdder")
      .def_static("add_vectors", &VectorAdder::add_vectors,
                  "Add two vectors using CUDA");
  m.def("get_data", &get_data, py::return_value_policy::move);
  m.def("process_data", &process_data);

  bind_dedispersed_data<uint8_t>(m, "DedisperedDataUint8");
  bind_dedispersed_data<uint16_t>(m, "DedisperedDataUint16");
  bind_dedispersed_data<uint32_t>(m, "DedisperedDataUint32");

  m.def("_dedisper_fil_uint16",
        py::overload_cast<std::string, float, float, float, float, float, int,
                          float, int>(&dedispered_fil<uint16_t>),
        py::arg("filename"), py::arg("dm_low"), py::arg("dm_high"),
        py::arg("freq_start"), py::arg("freq_end"), py::arg("dm_step") = 1,
        py::arg("time_downsample") = 64, py::arg("t_sample") = 0.5,
        py::arg("njobs") = 64);

  m.def("_dedisper_fil_uint8",
        py::overload_cast<std::string, float, float, float, float, float, int,
                          float, int>(&dedispered_fil<uint8_t>),
        py::arg("filename"), py::arg("dm_low"), py::arg("dm_high"),
        py::arg("freq_start"), py::arg("freq_end"), py::arg("dm_step") = 1,
        py::arg("time_downsample") = 64, py::arg("t_sample") = 0.5,
        py::arg("njobs") = 64);
}
