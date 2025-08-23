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

// Template specializations for dedispersion output types (precision scaling)
template<typename InputType>
struct DedispersionOutputType {};

template<>
struct DedispersionOutputType<uint8_t> {
    using type = uint32_t;  // 8-bit -> 32-bit to prevent overflow
};

template<>
struct DedispersionOutputType<uint16_t> {
    using type = uint32_t;  // 16-bit -> 32-bit to prevent overflow
};

template<>
struct DedispersionOutputType<uint32_t> {
    using type = uint64_t;  // 32-bit -> 64-bit to prevent overflow
};

// Convenient alias for dedispersion output type
template<typename T>
using dedispersion_output_t = typename DedispersionOutputType<T>::type;

// Template class for type-safe dedispersion data containers
template<typename OutputType>
struct DedispersedDataTyped {
  std::vector<std::shared_ptr<OutputType[]>> dm_times;
  std::vector<size_t> shape;
  int dm_ndata;
  int downtsample_ndata;
  float dm_low;
  float dm_high;
  float dm_step;
  float tsample;
  std::string filname;
  
  // Convert to uint64_t based structure for compatibility
dedisperseddata to_uint64() const {
    dedisperseddata result;
    result.shape = shape;
    result.dm_ndata = dm_ndata;
    result.downtsample_ndata = downtsample_ndata;
    result.dm_low = dm_low;
    result.dm_high = dm_high;
    result.dm_step = dm_step;
    result.tsample = tsample;
    result.filname = filname;
    
    // Convert typed arrays to uint64_t arrays
    result.dm_times.reserve(dm_times.size());
    for (const auto& typed_array : dm_times) {
      const size_t total_elements = dm_ndata * downtsample_ndata;
      auto uint64_array = std::shared_ptr<uint64_t[]>(
          new (std::align_val_t{4096}) uint64_t[total_elements](),
          [](uint64_t *p) { operator delete[](p, std::align_val_t{4096}); });
      
      // Type conversion loop
      for (size_t i = 0; i < total_elements; ++i) {
        uint64_array[i] = static_cast<uint64_t>(typed_array[i]);
      }
      
      result.dm_times.emplace_back(std::move(uint64_array));
    }
    
    return result;
  }
};

// Type aliases for specific precision levels
using DedispersedData8to16 = DedispersedDataTyped<uint16_t>;
using DedispersedData16to32 = DedispersedDataTyped<uint32_t>;
using DedispersedData32to64 = DedispersedDataTyped<uint64_t>;


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
                   : cv::INTER_LANCZOS4;
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

// Template version for typed dedispersion data
template<typename InputType>
inline dedisperseddata_uint8
preprocess_typed_dedisperseddata(const DedispersedDataTyped<dedispersion_output_t<InputType>>& in,
                                 int target_size = 512)
{
    omp_set_num_threads(32);
    /* ---------- 输入检查 ---------- */
    if (in.shape.size() < 2)
        throw std::runtime_error("DedispersedDataTyped.shape 长度必须 ≥ 2");
    const int src_rows = static_cast<int>(in.shape[0]);
    const int src_cols = static_cast<int>(in.shape[1]);
    if (src_rows <= 0 || src_cols <= 0)
        throw std::runtime_error("DedispersedDataTyped.shape 非法");

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
        using OutputType = dedispersion_output_t<InputType>;

        /* 1) typed data → float32  (并行像素拷贝) */
        cv::Mat1f src32(src_rows, src_cols);
        const OutputType* raw = frame_ptr.get();
#pragma omp parallel for schedule(static) collapse(2)
        for (int r = 0; r < src_rows; ++r)
            for (int c = 0; c < src_cols; ++c)
                src32(r, c) = static_cast<float>(raw[r * src_cols + c]);

        /* 2) resize 到 target_size×target_size */
        cv::Mat resized;
        int interp = (src_rows > target_size && src_cols > target_size)
                   ? cv::INTER_AREA
                   : cv::INTER_LANCZOS4;
        cv::resize(src32, resized, {target_size, target_size}, 0, 0, interp);

        /* 3) 归一化 0-255、灰度→RGB、Viridis */
        cv::Mat norm8u, rgb, viridis;
        cv::normalize(resized, norm8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::cvtColor(norm8u, rgb, cv::COLOR_GRAY2RGB);
        cv::applyColorMap(rgb, viridis, cv::COLORMAP_VIRIDIS);

        /* 4) 拷贝到 4096 对齐缓冲区 */
        const size_t bytes = viridis.total() * viridis.elemSize(); // target_size*target_size*3
        auto deleter = [](uint8_t* p){ operator delete[](p, std::align_val_t{4096}); };
        std::shared_ptr<uint8_t[]> buf(
            reinterpret_cast<uint8_t*>(operator new[](bytes, std::align_val_t{4096})),
            deleter);

        std::memcpy(buf.get(), viridis.data, bytes);
        out.dm_times[idx] = std::move(buf);
    }

    return out;
}

// Original function kept for compatibility
inline dedisperseddata_uint8
preprocess_dedisperseddata_original(const dedisperseddata& in,
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

inline dedisperseddata_uint8
preprocess_dedisperseddata_with_slicing(const dedisperseddata& in, Header header, 
                                        int time_downsample,
                                        float slice_duration = 0.5,  // 切片时长(秒)
                                        int target_size = 512)
{
    omp_set_num_threads(32);
    
    /* ---------- 输入检查 ---------- */
    if (in.shape.size() < 2)
        throw std::runtime_error("dedisperseddata.shape 长度必须 ≥ 2");
    if (in.dm_times.empty())
        throw std::runtime_error("dedisperseddata.dm_times 为空");
        
    const int src_rows = static_cast<int>(in.shape[0]);  // DM steps
    const int src_cols = static_cast<int>(in.shape[1]);  // Time samples (already downsampled)
    if (src_rows <= 0 || src_cols <= 0)
        throw std::runtime_error("dedisperseddata.shape 非法");

    printf("Input data shape: [%d, %d], original tsample: %.6f, time_downsample: %d\n", 
           src_rows, src_cols, header.tsamp, time_downsample);

    // 计算切片参数
    // 原始tsamp是单个时间样本的时间间隔
    // 下采样后，每个样本对应 time_downsample * header.tsamp 的时间
    const float downsampled_tsamp = header.tsamp * time_downsample;
    const float total_time = src_cols * downsampled_tsamp;  // 总的观测时间
    const int samples_per_slice = static_cast<int>(slice_duration / downsampled_tsamp);
    const int num_slices = (src_cols + samples_per_slice - 1) / samples_per_slice;
    
    printf("Downsampled tsamp: %.6f s, Total time: %.3f s, Samples per slice: %d, Number of slices: %d\n", 
           downsampled_tsamp, total_time, samples_per_slice, num_slices);

    /* ---------- 元数据填充 ---------- */
    dedisperseddata_uint8 out;
    out.dm_low   = in.dm_low;
    out.dm_high  = in.dm_high;
    out.dm_step  = in.dm_step;
    out.tsample  = slice_duration;  // 每个切片的时长
    out.filname  = in.filname;

    out.shape = {static_cast<size_t>(target_size),
                 static_cast<size_t>(target_size),
                 3};
    out.dm_ndata          = target_size;
    out.downtsample_ndata = target_size;

    /* ---------- 预分配输出容器 ---------- */
    out.dm_times.resize(num_slices);

    // 获取原始数据指针
    const uint64_t* raw_data = in.dm_times[0].get();

    /* ---------- OpenMP 并行处理切片 ---------- */
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t slice_idx = 0; slice_idx < static_cast<ptrdiff_t>(num_slices); ++slice_idx)
    {
        // 计算当前切片的时间范围
        const int start_col = slice_idx * samples_per_slice;
        const int end_col = std::min(start_col + samples_per_slice, src_cols);
        const int actual_slice_cols = end_col - start_col;

        // printf("Processing slice %td: columns [%d, %d), actual width: %d\n", 
        //        slice_idx, start_col, end_col, actual_slice_cols);

        /* 1) 提取切片并转换为 float32 */
        cv::Mat1f slice32(src_rows, actual_slice_cols);
#pragma omp parallel for schedule(static) collapse(2)
        for (int r = 0; r < src_rows; ++r) {
            for (int c = 0; c < actual_slice_cols; ++c) {
                const int source_col = start_col + c;
                slice32(r, c) = static_cast<float>(raw_data[r * src_cols + source_col]);
            }
        }

        /* 2) resize 到 512×512 */
        cv::Mat resized;
        int interp = (src_rows > target_size && actual_slice_cols > target_size)
                   ? cv::INTER_AREA
                   : cv::INTER_LANCZOS4;
        cv::resize(slice32, resized, {target_size, target_size}, 0, 0, interp);

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
        out.dm_times[slice_idx] = std::move(buf);
    }

    printf("Slicing preprocessing completed: %d slices generated\n", num_slices);
    return out;
}

// Template version for typed dedispersion data with slicing
template<typename InputType>
inline dedisperseddata_uint8
preprocess_typed_dedisperseddata_with_slicing(const DedispersedDataTyped<dedispersion_output_t<InputType>>& in, 
                                              Header header, 
                                              int time_downsample,
                                              float slice_duration = 0.5,  // 切片时长(秒)
                                              int target_size = 512)
{
    omp_set_num_threads(32);
    
    /* ---------- 输入检查 ---------- */
    if (in.shape.size() < 2)
        throw std::runtime_error("DedispersedDataTyped.shape 长度必须 ≥ 2");
    if (in.dm_times.empty())
        throw std::runtime_error("DedispersedDataTyped.dm_times 为空");
        
    const size_t src_rows = in.shape[0];  // DM steps
    const size_t src_cols = in.shape[1];  // Time samples (already downsampled)
    if (src_rows == 0 || src_cols == 0)
        throw std::runtime_error("DedispersedDataTyped.shape 非法");

    printf("Input typed data shape: [%zu, %zu], original tsample: %.6f \n", 
           src_rows, src_cols, header.tsamp);

    // 计算切片参数
    const float downsampled_tsamp = header.tsamp * time_downsample;
    const float total_time = static_cast<float>(src_cols) * downsampled_tsamp;
    const size_t samples_per_slice = static_cast<size_t>(slice_duration / downsampled_tsamp);
    const size_t num_slices = (src_cols + samples_per_slice - 1) / samples_per_slice;
    
    printf("Typed data - Downsampled tsamp: %.6f s, Total time: %.3f s, Samples per slice: %zu, Number of slices: %zu\n", 
           downsampled_tsamp, total_time, samples_per_slice, num_slices);

    /* ---------- 元数据填充 ---------- */
    dedisperseddata_uint8 out;
    out.dm_low   = in.dm_low;
    out.dm_high  = in.dm_high;
    out.dm_step  = in.dm_step;
    out.tsample  = slice_duration;  // 每个切片的时长
    out.filname  = in.filname;

    out.shape = {static_cast<size_t>(target_size),
                 static_cast<size_t>(target_size),
                 3};
    out.dm_ndata          = target_size;
    out.downtsample_ndata = target_size;

    /* ---------- 预分配输出容器 ---------- */
    out.dm_times.resize(num_slices);

    // 获取原始数据指针
    using OutputType = dedispersion_output_t<InputType>;
    const OutputType* raw_data = in.dm_times[0].get();

    /* ---------- OpenMP 并行处理切片 ---------- */
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t slice_idx = 0; slice_idx < static_cast<ptrdiff_t>(num_slices); ++slice_idx)
    {
        // 计算当前切片的时间范围
        const size_t start_col = static_cast<size_t>(slice_idx) * samples_per_slice;
        const size_t end_col = std::min(start_col + samples_per_slice, src_cols);
        const size_t actual_slice_cols = end_col - start_col;

        /* 1) 提取切片并转换为 float32 */
        cv::Mat1f slice32(static_cast<int>(src_rows), static_cast<int>(actual_slice_cols));
#pragma omp parallel for schedule(static) collapse(2)
        for (size_t r = 0; r < src_rows; ++r) {
            for (size_t c = 0; c < actual_slice_cols; ++c) {
                const size_t source_col = start_col + c;
                slice32(static_cast<int>(r), static_cast<int>(c)) = static_cast<float>(raw_data[r * src_cols + source_col]);
            }
        }

        /* 2) resize 到 target_size×target_size */
        cv::Mat resized;
        int interp = (static_cast<int>(src_rows) > target_size && static_cast<int>(actual_slice_cols) > target_size)
                   ? cv::INTER_AREA
                   : cv::INTER_LANCZOS4;
        cv::resize(slice32, resized, {target_size, target_size}, 0, 0, interp);

        /* 3) 归一化 0-255、灰度→RGB、Viridis */
        cv::Mat norm8u, rgb, viridis;
        cv::normalize(resized, norm8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::cvtColor(norm8u, rgb, cv::COLOR_GRAY2RGB);
        cv::applyColorMap(rgb, viridis, cv::COLORMAP_VIRIDIS);

        /* 4) 拷贝到 4096 对齐缓冲区 */
        const size_t bytes = viridis.total() * viridis.elemSize(); // target_size*target_size*3
        auto deleter = [](uint8_t* p){ operator delete[](p, std::align_val_t{4096}); };
        std::shared_ptr<uint8_t[]> buf(
            reinterpret_cast<uint8_t*>(operator new[](bytes, std::align_val_t{4096})),
            deleter);

        std::memcpy(buf.get(), viridis.data, bytes);
        out.dm_times[slice_idx] = std::move(buf);
    }

    printf("Typed slicing preprocessing completed: %zu slices generated\n", num_slices);
    return out;
}

// Original function kept for compatibility
inline dedisperseddata_uint8
preprocess_dedisperseddata_with_slicing_original(const dedisperseddata& in, Header header, 
                                        int time_downsample,
                                        float slice_duration = 0.5,  // 切片时长(秒)
                                        int target_size = 512)
{
    omp_set_num_threads(32);
    
    /* ---------- 输入检查 ---------- */
    if (in.shape.size() < 2)
        throw std::runtime_error("dedisperseddata.shape 长度必须 ≥ 2");
    if (in.dm_times.empty())
        throw std::runtime_error("dedisperseddata.dm_times 为空");
        
    const int src_rows = static_cast<int>(in.shape[0]);  // DM steps
    const int src_cols = static_cast<int>(in.shape[1]);  // Time samples (already downsampled)
    if (src_rows <= 0 || src_cols <= 0)
        throw std::runtime_error("dedisperseddata.shape 非法");

    printf("Input data shape: [%d, %d], original tsample: %.6f, time_downsample: %d\n", 
           src_rows, src_cols, header.tsamp, time_downsample);

    // 计算切片参数
    // 原始tsamp是单个时间样本的时间间隔
    // 下采样后，每个样本对应 time_downsample * header.tsamp 的时间
    const float downsampled_tsamp = header.tsamp * time_downsample;
    const float total_time = src_cols * downsampled_tsamp;  // 总的观测时间
    const int samples_per_slice = static_cast<int>(slice_duration / downsampled_tsamp);
    const int num_slices = (src_cols + samples_per_slice - 1) / samples_per_slice;
    
    printf("Downsampled tsamp: %.6f s, Total time: %.3f s, Samples per slice: %d, Number of slices: %d\n", 
           downsampled_tsamp, total_time, samples_per_slice, num_slices);

    /* ---------- 元数据填充 ---------- */
    dedisperseddata_uint8 out;
    out.dm_low   = in.dm_low;
    out.dm_high  = in.dm_high;
    out.dm_step  = in.dm_step;
    out.tsample  = slice_duration;  // 每个切片的时长
    out.filname  = in.filname;

    out.shape = {static_cast<size_t>(target_size),
                 static_cast<size_t>(target_size),
                 3};
    out.dm_ndata          = target_size;
    out.downtsample_ndata = target_size;

    /* ---------- 预分配输出容器 ---------- */
    out.dm_times.resize(num_slices);

    // 获取原始数据指针
    const uint64_t* raw_data = in.dm_times[0].get();

    /* ---------- OpenMP 并行处理切片 ---------- */
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t slice_idx = 0; slice_idx < static_cast<ptrdiff_t>(num_slices); ++slice_idx)
    {
        // 计算当前切片的时间范围
        const int start_col = slice_idx * samples_per_slice;
        const int end_col = std::min(start_col + samples_per_slice, src_cols);
        const int actual_slice_cols = end_col - start_col;

        // printf("Processing slice %td: columns [%d, %d), actual width: %d\n", 
        //        slice_idx, start_col, end_col, actual_slice_cols);

        /* 1) 提取切片并转换为 float32 */
        cv::Mat1f slice32(src_rows, actual_slice_cols);
#pragma omp parallel for schedule(static) collapse(2)
        for (int r = 0; r < src_rows; ++r) {
            for (int c = 0; c < actual_slice_cols; ++c) {
                const int source_col = start_col + c;
                slice32(r, c) = static_cast<float>(raw_data[r * src_cols + source_col]);
            }
        }

        /* 2) resize 到 512×512 */
        cv::Mat resized;
        int interp = (src_rows > target_size && actual_slice_cols > target_size)
                   ? cv::INTER_AREA
                   : cv::INTER_LINEAR;
        cv::resize(slice32, resized, {target_size, target_size}, 0, 0, interp);

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
        out.dm_times[slice_idx] = std::move(buf);
    }

    printf("Slicing preprocessing completed: %d slices generated\n", num_slices);
    return out;
}


#endif //_DATA_H
