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

PYBIND11_MODULE(_astroflow_core, m) {

  bind_dedispersed_data<uint64_t>(m, "DedisperedData");
  bind_dedispersed_data_uint8(m, "DedispersedDataUint8");
  bind_filterbank(m);
  bind_spectrum<uint8_t>(m, "Spectrum8");
  bind_spectrum<uint16_t>(m, "Spectrum16");
  bind_spectrum<uint32_t>(m, "Spectrum32");
  bind_header(m);

  m.def("_dedispered_fil", &dedispered_fil, py::arg("filename"),
        py::arg("dm_low"), py::arg("dm_high"), py::arg("freq_start"),
        py::arg("freq_end"), py::arg("dm_step") = 1,
        py::arg("time_downsample") = 2, py::arg("t_sample") = 0.5,
        py::arg("target") = GPU_TARGET,py::arg("mask_file") = "mask.txt");

  m.def("_dedispered_fil_with_dm_uint8",
        &cpucal::dedispered_fil_with_dm<uint8_t>, py::arg("fil"),
        py::arg("tstart"), py::arg("tend"), py::arg("dm"),
        py::arg("freq_start"), py::arg("freq_end"), py::arg("maskfile"));
  m.def("_dedispered_fil_with_dm_uint16",
        &cpucal::dedispered_fil_with_dm<uint16_t>, py::arg("fil"),
        py::arg("tstart"), py::arg("tend"), py::arg("dm"),
        py::arg("freq_start"), py::arg("freq_end"), py::arg("maskfile"));
  m.def("_dedispered_fil_with_dm_uint32",
        &cpucal::dedispered_fil_with_dm<uint32_t>, py::arg("fil"),
        py::arg("tstart"), py::arg("tend"), py::arg("dm"),
        py::arg("freq_start"), py::arg("freq_end"), py::arg("maskfile"));
  m.def("_dedisperse_spec", &dedisperse_spec_py<uint8_t>, py::arg("data"),
        py::arg("header"), py::arg("dm_low"), py::arg("dm_high"),
        py::arg("freq_start"), py::arg("dm_step") = 1,
        py::arg("ref_freq") = REF_FREQ_END, py::arg("time_downsample") = 2,
        py::arg("t_sample") = 0.5, py::arg("mask_file") = "mask.txt");
  m.def("_dedisperse_spec", &dedisperse_spec_py<uint16_t>, py::arg("data"),
        py::arg("header"), py::arg("dm_low"), py::arg("dm_high"),
        py::arg("freq_start"), py::arg("dm_step") = 1,
        py::arg("ref_freq") = REF_FREQ_END, py::arg("time_downsample") = 2,
        py::arg("t_sample") = 0.5, py::arg("mask_file") = "mask.txt");
  m.def("_dedisperse_spec", &dedisperse_spec_py<uint32_t>, py::arg("data"),
        py::arg("header"), py::arg("dm_low"), py::arg("dm_high"),
        py::arg("freq_start"), py::arg("dm_step") = 1,
        py::arg("ref_freq") = REF_FREQ_END, py::arg("time_downsample") = 2,
        py::arg("t_sample") = 0.5, py::arg("mask_file") = "mask.txt");

  m.def("_dedisperse_spec_with_dm", &dedisperse_spec_with_dm_py<uint8_t>,
        py::arg("data"), py::arg("header"), py::arg("tstart"), py::arg("dm"),
        py::arg("tend"), py::arg("freq_start"), py::arg("freq_end"), py::arg("maskfile"));
  m.def("_dedisperse_spec_with_dm", &dedisperse_spec_with_dm_py<uint16_t>,
        py::arg("data"), py::arg("header"), py::arg("tstart"), py::arg("dm"),
        py::arg("tend"), py::arg("freq_start"), py::arg("freq_end"), py::arg("maskfile"));
  m.def("_dedisperse_spec_with_dm", &dedisperse_spec_with_dm_py<uint32_t>,
        py::arg("data"), py::arg("header"), py::arg("tstart"), py::arg("dm"),
        py::arg("tend"), py::arg("freq_start"), py::arg("freq_end"), py::arg("maskfile"));
}
