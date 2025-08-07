/**
 * Project astroflow
 */

#ifndef _DATA_H
#define _DATA_H

#include <algorithm>
#include <cstdint>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <omp.h>
#include <vector>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

struct Header {
  float mjd;
  float tsamp;
  float fch1;
  float foff;
  int nbits;
  int nchans;
  long int ndata;
  std::string filename;

  Header()
      : mjd(0), tsamp(0), fch1(0), foff(0), nbits(0), nchans(0), ndata(0) {}
  Header(float mjd, float tsamp, float fch1, float foff, int nbits, int nchans,
         long int ndata, std::string filename)
      : mjd(mjd), tsamp(tsamp), fch1(fch1), foff(foff), nbits(nbits),
        nchans(nchans), ndata(ndata), filename(filename) {}
  Header(py::object header) {
    filename = header.attr("filename").cast<std::string>();
    mjd = header.attr("mjd").cast<float>();
    tsamp = header.attr("tsamp").cast<float>();
    fch1 = header.attr("fch1").cast<float>();
    foff = header.attr("foff").cast<float>();
    nbits = header.attr("nbits").cast<int>();
    nchans = header.attr("nchans").cast<int>();
    ndata = header.attr("ndata").cast<long int>();
  }
};

struct dedisperseddata_uint8 {
  std::vector<std::shared_ptr<uint8_t[]>> dm_times;

  std::vector<size_t> shape;

  int dm_ndata;            // = H
  int downtsample_ndata;   // = W

  float dm_low;
  float dm_high;
  float dm_step;
  float tsample;
  std::string filname;
};




struct dedisperseddata {
  std::vector<std::shared_ptr<uint64_t[]>> dm_times;

  //[DM,downtsample]
  std::vector<size_t> shape;
  int dm_ndata;
  int downtsample_ndata;

  // dm_low, dm_high, dm_step, dm_size
  float dm_low;
  float dm_high;
  float dm_step;
  float tsample;
  // filname
  std::string filname;
};

template <typename T> struct Spectrum {
  std::shared_ptr<T[]> data;
  size_t nchans;
  size_t ntimes;
  float tstart;
  float tend;
  float dm;
  int nbits;
  float freq_start;
  float freq_end;
};

class DynamicArray {
public:
  enum class Type { U8, U16, U32 };

  // 构造函数集
  DynamicArray(const uint8_t *data, size_t size) { init(data, size); }
  DynamicArray(const uint16_t *data, size_t size) { init(data, size); }
  DynamicArray(const uint32_t *data, size_t size) { init(data, size); }

  class Proxy {
  public:
    Proxy(DynamicArray &arr, size_t index) : arr_(arr), index_(index) {}

    operator uint32_t() const {
      switch (arr_.type_) {
      case Type::U8:
        return static_cast<uint8_t *>(arr_.data_.get())[index_];
      case Type::U16:
        return static_cast<uint16_t *>(arr_.data_.get())[index_];
      case Type::U32:
        return static_cast<uint32_t *>(arr_.data_.get())[index_];
      default:
        throw std::runtime_error("Invalid type");
      }
    }

    Proxy &operator=(uint32_t value) {
      arr_.ensure_unique();
      switch (arr_.type_) {
      case Type::U8:
        static_cast<uint8_t *>(arr_.data_.get())[index_] = value & 0xFF;
        break;
      case Type::U16:
        static_cast<uint16_t *>(arr_.data_.get())[index_] = value & 0xFFFF;
        break;
      case Type::U32:
        static_cast<uint32_t *>(arr_.data_.get())[index_] = value;
        break;
      }
      return *this;
    }

  private:
    DynamicArray &arr_;
    size_t index_;
  };

  uint32_t operator[](size_t index) const {
    if (index >= size_)
      throw std::out_of_range("Index out of range");
    switch (type_) {
    case Type::U8:
      return static_cast<uint8_t *>(data_.get())[index];
    case Type::U16:
      return static_cast<uint16_t *>(data_.get())[index];
    case Type::U32:
      return static_cast<uint32_t *>(data_.get())[index];
    default:
      throw std::runtime_error("Invalid type");
    }
  }

  Proxy operator[](size_t index) {
    if (index >= size_)
      throw std::out_of_range("Index out of range");
    return Proxy(*this, index);
  }

  DynamicArray(const DynamicArray &) = default;
  DynamicArray &operator=(const DynamicArray &) = default;

  size_t size() const { return size_; }

private:
  template <typename T> void init(const T *data, size_t size) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
                  "Unsupported type");

    set_type<T>();
    size_ = size;
    T *buffer = new T[size];
    // std::copy(data, data + size, buffer);
    memcpy(buffer, data, size * sizeof(T));
    data_ = std::shared_ptr<void>(
        buffer, [](void *ptr) { delete[] static_cast<T *>(ptr); });
  }

  template <typename T> void set_type() {
    if constexpr (std::is_same_v<T, uint8_t>)
      type_ = Type::U8;
    else if constexpr (std::is_same_v<T, uint16_t>)
      type_ = Type::U16;
    else
      type_ = Type::U32;
  }

  void ensure_unique() {
    if (!data_.unique()) {
      switch (type_) {
      case Type::U8:
        recreate<uint8_t>();
        break;
      case Type::U16:
        recreate<uint16_t>();
        break;
      case Type::U32:
        recreate<uint32_t>();
        break;
      }
    }
  }

  template <typename T> void recreate() {
    T *old_data = static_cast<T *>(data_.get());
    T *new_data = new T[size_];
    std::copy(old_data, old_data + size_, new_data);
    data_ = std::shared_ptr<void>(
        new_data, [](void *ptr) { delete[] static_cast<T *>(ptr); });
  }

  Type type_;
  size_t size_ = 0;
  std::shared_ptr<void> data_;

  friend class Proxy;
};


inline dedisperseddata_uint8
preprocess_dedisperseddata(const dedisperseddata& in,
                           int target_size = 512)
{
    omp_set_num_threads(32);
    /* ---------- 输入检查 ---------- */
    if (in.shape.size() < 2)
        throw std::runtime_error("dedisperseddata.shape 长度必须 ≥ 2");
    const int src_rows = static_cast<int>(in.shape[0]);
    const int src_cols = static_cast<int>(in.shape[1]);
    if (src_rows <= 0 || src_cols <= 0)
        throw std::runtime_error("dedisperseddata.shape 非法");

    /* ---------- 元数据填充 ---------- */
    dedisperseddata_uint8 out;
    out.dm_low   = in.dm_low;
    out.dm_high  = in.dm_high;
    out.dm_step  = in.dm_step;
    out.tsample  = in.tsample;
    out.filname  = in.filname;

    out.shape = {static_cast<size_t>(target_size),
                 static_cast<size_t>(target_size),
                 3};
    out.dm_ndata          = target_size;
    out.downtsample_ndata = target_size;

    /* ---------- 预分配输出容器 ---------- */
    const size_t n_frames = in.dm_times.size();
    out.dm_times.resize(n_frames);

    /* ---------- OpenMP 并行处理 ---------- */
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(n_frames); ++idx)
    {
        const auto& frame_ptr = in.dm_times[idx];

        /* 1) uint64 → float32  (并行像素拷贝) */
        cv::Mat1f src32(src_rows, src_cols);
        const uint64_t* raw = frame_ptr.get();
#pragma omp parallel for schedule(static) collapse(2)
        for (int r = 0; r < src_rows; ++r)
            for (int c = 0; c < src_cols; ++c)
                src32(r, c) = static_cast<float>(raw[r * src_cols + c]);

        /* 2) resize 到 512×512 */
        cv::Mat resized;
        int interp = (src_rows > target_size && src_cols > target_size)
                   ? cv::INTER_AREA
                   : cv::INTER_LINEAR;
        cv::resize(src32, resized, {target_size, target_size}, 0, 0, interp);

        /* 3) 归一化 0-255、灰度→RGB、Viridis */
        cv::Mat norm8u, rgb, viridis;
        cv::normalize(resized, norm8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::cvtColor(norm8u, rgb, cv::COLOR_GRAY2RGB);
        cv::applyColorMap(rgb, viridis, cv::COLORMAP_VIRIDIS);

        /* 4) 拷贝到 4096 对齐缓冲区 */
        const size_t bytes = viridis.total() * viridis.elemSize(); // 512*512*3
        auto deleter = [](uint8_t* p){ operator delete[](p, std::align_val_t{4096}); };
        std::shared_ptr<uint8_t[]> buf(
            reinterpret_cast<uint8_t*>(operator new[](bytes, std::align_val_t{4096})),
            deleter);

        std::memcpy(buf.get(), viridis.data, bytes);
        out.dm_times[idx] = std::move(buf);
    }

    return out;
}


#endif //_DATA_H
