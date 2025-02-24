#include "cpp_code.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// 使用make_shared风格创建共享数组
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

PYBIND11_MODULE(demo, m) {
  py::class_<VectorAdder>(m, "VectorAdder")
      .def_static("add_vectors", &VectorAdder::add_vectors,
                  "Add two vectors using CUDA");
  m.def("get_data", &get_data, py::return_value_policy::move);
  m.def("process_data", &process_data);
}
