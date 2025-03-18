#include "cpp_code.h"
#include "pyapi.h"
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
  using Data = dedisperseddata;
  py::class_<Data>(m, class_name)
      .def(py::init<>())
      .def_property_readonly("dm_times",
                             [](const Data &data) -> py::object {
                               std::vector<py::array_t<T>> py_arrays;
                               for (const auto &ptr : data.dm_times) {
                                 // vector<size_t> shape = {data.shape[0],
                                 // data.shape[1]}; vector<size_t> strides =
                                 // {data.shape[1], 1};
                                 vector<size_t> shape = {data.shape[0] *
                                                         data.shape[1]};
                                 auto arr = py::array_t<T>(shape, ptr.get());
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

void bind_filterbank(py::module &m) {
  py::class_<Filterbank>(m, "Filterbank")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("info", &Filterbank::info)
      .def_readonly("filename", &Filterbank::filename)
      .def_readonly("nchans", &Filterbank::nchans)
      .def_readonly("nifs", &Filterbank::nifs)
      .def_readonly("nbits", &Filterbank::nbits)
      .def_readonly("fch1", &Filterbank::fch1)
      .def_readonly("foff", &Filterbank::foff)
      .def_readonly("tstart", &Filterbank::tstart)
      .def_readonly("tsamp", &Filterbank::tsamp)
      .def_readonly("ndata", &Filterbank::ndata)
      .def_property_readonly("data", [](const Filterbank &fil) -> py::object {
        int nbits = fil.nbits;
        int nifs = fil.nifs;
        int nchans = fil.nchans;
        switch (nbits) {
        case 8: {
          uint8_t *data8 = static_cast<uint8_t *>(fil.data);
          vector<size_t> shape = {static_cast<size_t>(fil.ndata),
                                  static_cast<size_t>(nifs),
                                  static_cast<size_t>(nchans)};
          vector<size_t> strides = {static_cast<size_t>(nifs * nchans),
                                    static_cast<size_t>(nchans), 1};
          auto arr = py::array_t<uint8_t>(shape, strides, data8);
          return arr;
        }
        case 16: {
          uint16_t *data16 = static_cast<uint16_t *>(fil.data);
          vector<size_t> shape = {static_cast<size_t>(fil.ndata),
                                  static_cast<size_t>(nifs),
                                  static_cast<size_t>(nchans)};
          vector<size_t> strides = {static_cast<size_t>(nifs * nchans),
                                    static_cast<size_t>(nchans), 1};
          auto arr = py::array_t<uint16_t>(shape, strides, data16);
          return arr;
        }
        case 32: {
          uint32_t *data32 = static_cast<uint32_t *>(fil.data);
          vector<size_t> shape = {static_cast<size_t>(fil.ndata),
                                  static_cast<size_t>(nifs),
                                  static_cast<size_t>(nchans)};
          vector<size_t> strides = {static_cast<size_t>(nifs * nchans),
                                    static_cast<size_t>(nchans), 1};
          auto arr = py::array_t<uint32_t>(shape, strides, data32);
          return arr;
        }
        default: {
          throw py::value_error("Unsupported nbits value: " +
                                std::to_string(nbits));
        }
        }
      });
}

PYBIND11_MODULE(_astroflow_core, m) {

  py::class_<VectorAdder>(m, "VectorAdder")
      .def_static("add_vectors", &VectorAdder::add_vectors,
                  "Add two vectors using CUDA");
  m.def("get_data", &get_data, py::return_value_policy::move);
  m.def("process_data", &process_data);

  bind_dedispersed_data<uint32_t>(m, "DedisperedData");
  bind_filterbank(m);

  m.def("_dedispered_fil", &dedispered_fil, py::arg("filename"),
        py::arg("dm_low"), py::arg("dm_high"), py::arg("freq_start"),
        py::arg("freq_end"), py::arg("dm_step") = 1,
        py::arg("time_downsample") = 2, py::arg("t_sample") = 0.5,
        py::arg("target") = GPU_TARGET);
}
