#include "pyapi.h"
#include "data.h"

namespace py = pybind11;

dedisperseddata dedispered_fil(std::string filename, float dm_low,
                               float dm_high, float freq_start, float freq_end,
                               float dm_step, int time_downsample,
                               float t_sample, int target) {
  Filterbank fil(filename);
  fil.info();
  switch (fil.nbits) {
  case 8: {
    if (target == GPU_TARGET) {
      return gpucal::dedispered_fil_cuda<uint8_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    } else if (target == CPU_TARGET) {
      return cpucal::dedispered_fil_omp<uint8_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    }
  }
  case 16: {
    if (target == GPU_TARGET) {
      return gpucal::dedispered_fil_cuda<uint16_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    } else if (target == CPU_TARGET) {
      return cpucal::dedispered_fil_omp<uint16_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    }
  }
  case 32: {
    if (target == GPU_TARGET) {
      return gpucal::dedispered_fil_cuda<uint32_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    } else if (target == CPU_TARGET) {
      return cpucal::dedispered_fil_omp<uint32_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);
    }
  }
  default: {
    throw std::runtime_error("Unsupported nbits value: " +
                             std::to_string(fil.nbits));
  }
  }
};

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

template void bind_dedispersed_data<uint64_t>(py::module &m,
                                              const char *class_name);

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

template void bind_spectrum<uint8_t>(py::module &m, const char *class_name);
template void bind_spectrum<uint16_t>(py::module &m, const char *class_name);
template void bind_spectrum<uint32_t>(py::module &m, const char *class_name);

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

void bind_header(py::module &m) {
  py::class_<Header>(m, "Header")
      .def(py::init<>())
      .def(py::init<float, float, float, float, int, int, long int,
                    std::string>())
      .def(py::init<py::object>())
      .def_readonly("mjd", &Header::mjd)
      .def_readonly("tsamp", &Header::tsamp)
      .def_readonly("fch1", &Header::fch1)
      .def_readonly("foff", &Header::foff)
      .def_readonly("nbits", &Header::nbits)
      .def_readonly("nchans", &Header::nchans)
      .def_readonly("ndata", &Header::ndata)
      .def_readonly("filename", &Header::filename);
}

template <typename T>
dedisperseddata
dedisperse_spec_py(py::array_t<T> data, Header header, float dm_low,
                   float dm_high, float freq_start, float freq_end,
                   float dm_step, int time_downsample, float t_sample) {

  auto data_ptr = data.request();
  auto data_ptr_ptr = static_cast<T *>(data_ptr.ptr);

  if (data_ptr.ndim != 1) {
    throw std::runtime_error("data must be 1D");
  }

  return gpucal::dedisperse_spec<T>(data_ptr_ptr, header, dm_low, dm_high,
                                    freq_start, freq_end, dm_step, REF_FREQ_END,
                                    time_downsample, t_sample);
}

template dedisperseddata
dedisperse_spec_py<uint8_t>(py::array_t<uint8_t> data, Header header,
                            float dm_low, float dm_high, float freq_start,
                            float freq_end, float dm_step, int time_downsample,
                            float t_sample);

template dedisperseddata
dedisperse_spec_py<uint16_t>(py::array_t<uint16_t> data, Header header,
                             float dm_low, float dm_high, float freq_start,
                             float freq_end, float dm_step, int time_downsample,
                             float t_sample);

template dedisperseddata
dedisperse_spec_py<uint32_t>(py::array_t<uint32_t> data, Header header,
                             float dm_low, float dm_high, float freq_start,
                             float freq_end, float dm_step, int time_downsample,
                             float t_sample);

template <typename T>
Spectrum<T> dedisperse_spec_with_dm_py(py::array_t<T> data, Header header,
                                       float tstart, float tend, float dm,
                                       float freq_start, float freq_end) {
  auto data_ptr = data.request();
  T *data_ptr_ptr = static_cast<T *>(data_ptr.ptr);

  if (data_ptr.ndim != 1) {
    throw std::runtime_error("data must be 1D");
  }

  return cpucal::dedisperse_spec_with_dm<T>(data_ptr_ptr, header, dm, tstart,
                                            tend, freq_start, freq_end);
}

template Spectrum<uint8_t>
dedisperse_spec_with_dm_py<uint8_t>(py::array_t<uint8_t> data, Header header,
                                    float dm, float tstart, float tend,
                                    float freq_start, float freq_end);

template Spectrum<uint16_t>
dedisperse_spec_with_dm_py<uint16_t>(py::array_t<uint16_t> data, Header header,
                                     float dm, float tstart, float tend,
                                     float freq_start, float freq_end);

template Spectrum<uint32_t>
dedisperse_spec_with_dm_py<uint32_t>(py::array_t<uint32_t> data, Header header,
                                     float dm, float tstart, float tend,
                                     float freq_start, float freq_end);
