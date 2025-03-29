#include "cpp_code.h"
#include "data.h"
#include "filterbank.h"
#include "pyapi.h"
#include <cstdint>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename T>
void bind_dedispersed_data(py::module &m, const char *class_name) {
  using Data = dedisperseddata;
  py::class_<Data>(m, class_name)
      .def(py::init<>())
      .def_property_readonly(
          "dm_times",
          [](const Data &data) -> py::object {
            std::vector<py::array_t<T>> py_arrays;
            for (const auto &ptr : data.dm_times) {
              auto capsule =
                  py::capsule(new std::shared_ptr<T[]>(ptr), [](void *p) {
                    delete static_cast<std::shared_ptr<T[]> *>(p);
                  });
              vector<size_t> shape = {data.shape[0] * data.shape[1]};
              vector<size_t> strides = {sizeof(T)};
              auto arr = py::array_t<T>(shape, strides, ptr.get(), capsule);
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
template <typename T>
void bind_spectrum(py::module &m, const char *class_name) {
  py::class_<Spectrum<T>>(m, class_name)
      .def(py::init<>())
      .def_property_readonly(
          "data",
          [](const Spectrum<T> &spec) -> py::object {
            auto data_ptr = spec.data;

            if (data_ptr.get() == nullptr) {
              throw std::runtime_error("Spectrum data is null");
            }

            auto capsule =
                py::capsule(new std::shared_ptr<T[]>(data_ptr), [](void *p) {
                  delete static_cast<std::shared_ptr<T[]> *>(p);
                });

            vector<size_t> shape = {static_cast<size_t>(spec.ntimes),
                                    static_cast<size_t>(spec.nchans)};

            vector<size_t> strides = {sizeof(T) * spec.nchans, sizeof(T)};

            return py::array_t<T>(shape, strides, data_ptr.get(), capsule);
          })
      .def_readonly("ntimes", &Spectrum<T>::ntimes)
      .def_readonly("nchans", &Spectrum<T>::nchans)
      .def_readonly("tstart", &Spectrum<T>::tstart)
      .def_readonly("tend", &Spectrum<T>::tend)
      .def_readonly("dm", &Spectrum<T>::dm)
      .def_readonly("nbits", &Spectrum<T>::nbits)
      .def_readonly("freq_start", &Spectrum<T>::freq_start)
      .def_readonly("freq_end", &Spectrum<T>::freq_end);
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
      .def_property_readonly("data", [](Filterbank &fil) -> py::object {
        int nbits = fil.nbits;
        int nifs = fil.nifs;
        int nchans = fil.nchans;
        switch (nbits) {
        case 8: {
          std::shared_ptr<uint8_t[]> data_ptr(
              fil.get_shared_ptr_data<uint8_t>());

          auto capsule = py::capsule(
              new std::shared_ptr<uint8_t[]>(data_ptr), [](void *p) {
                delete static_cast<std::shared_ptr<uint8_t[]> *>(p);
              });

          vector<size_t> shape = {static_cast<size_t>(fil.ndata),
                                  static_cast<size_t>(nifs),
                                  static_cast<size_t>(nchans)};
          vector<size_t> strides = {
              static_cast<size_t>(nifs * nchans) * sizeof(uint8_t),
              static_cast<size_t>(nchans) * sizeof(uint8_t), sizeof(uint8_t)};
          return py::array_t<uint8_t>(shape, strides, data_ptr.get(), capsule);
        }
        case 16: {
          auto data_ptr = fil.get_shared_ptr_data<uint16_t>();
          auto capsule = py::capsule(new auto(data_ptr), [](void *p) {
            delete static_cast<std::shared_ptr<uint16_t[]> *>(p);
          });

          vector<size_t> shape = {static_cast<size_t>(fil.ndata),
                                  static_cast<size_t>(nifs),
                                  static_cast<size_t>(nchans)};
          vector<size_t> strides = {
              static_cast<size_t>(nifs * nchans) * sizeof(uint16_t),
              static_cast<size_t>(nchans) * sizeof(uint16_t), sizeof(uint16_t)};
          return py::array_t<uint16_t>(shape, strides, data_ptr.get(), capsule);
        }
        case 32: {
          auto data_ptr = fil.get_shared_ptr_data<uint32_t>();
          auto capsule = py::capsule(new auto(data_ptr), [](void *p) {
            delete static_cast<std::shared_ptr<uint32_t[]> *>(p);
          });
          vector<size_t> shape = {static_cast<size_t>(fil.ndata),
                                  static_cast<size_t>(nifs),
                                  static_cast<size_t>(nchans)};
          vector<size_t> strides = {
              static_cast<size_t>(nifs * nchans) * sizeof(uint32_t),
              static_cast<size_t>(nchans) * sizeof(uint32_t), sizeof(uint32_t)};
          return py::array_t<uint32_t>(shape, strides, data_ptr.get(), capsule);
        }
        default: {
          throw py::value_error("Unsupported nbits value: " +
                                std::to_string(nbits));
        }
        }
      });
}

PYBIND11_MODULE(_astroflow_core, m) {

  bind_dedispersed_data<uint32_t>(m, "DedisperedData");
  bind_filterbank(m);
  bind_spectrum<uint8_t>(m, "Spectrum8");
  bind_spectrum<uint16_t>(m, "Spectrum16");
  bind_spectrum<uint32_t>(m, "Spectrum32");

  m.def("_dedispered_fil", &dedispered_fil, py::arg("filename"),
        py::arg("dm_low"), py::arg("dm_high"), py::arg("freq_start"),
        py::arg("freq_end"), py::arg("dm_step") = 1,
        py::arg("time_downsample") = 2, py::arg("t_sample") = 0.5,
        py::arg("target") = GPU_TARGET);

  m.def("_dedispered_fil_with_dm_uint8",
        &cpucal::dedispered_fil_with_dm<uint8_t>, py::arg("fil"),
        py::arg("tstart"), py::arg("tend"), py::arg("dm"),
        py::arg("freq_start"), py::arg("freq_end"));
  m.def("_dedispered_fil_with_dm_uint16",
        &cpucal::dedispered_fil_with_dm<uint16_t>, py::arg("fil"),
        py::arg("tstart"), py::arg("tend"), py::arg("dm"),
        py::arg("freq_start"), py::arg("freq_end"));
  m.def("_dedispered_fil_with_dm_uint32",
        &cpucal::dedispered_fil_with_dm<uint32_t>, py::arg("fil"),
        py::arg("tstart"), py::arg("tend"), py::arg("dm"),
        py::arg("freq_start"), py::arg("freq_end"));
}
