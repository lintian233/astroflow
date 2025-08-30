#ifndef _RFIMARKER_CPU_H
#define _RFIMARKER_CPU_H

#include <vector>
#include <string>
#include <cstdint>
#include <iqrm.hpp>

/**
 * CPU RFI marker:
 * - load_mask(): 在 CPU 读入坏道列表
 * - mark_rfi(): 传入 host 指针 (h_data)，在 CPU 上把坏道通道清零
 *
 * 数据布局假设为: data[sample * num_channels + chan]  (row-major: time-major)
 */
template <typename T>
class RfiMarkerCPU {
public:
    RfiMarkerCPU();
    explicit RfiMarkerCPU(const char* mask_file);
    explicit RfiMarkerCPU(const std::string& mask_file) : RfiMarkerCPU(mask_file.c_str()) {}
    ~RfiMarkerCPU() = default;

    // 在 CPU 上将坏道置零；h_data 必须是 host 指针
    void mark_rfi(T* h_data,
                  unsigned int num_channels,
                  unsigned int num_samples);

    void load_mask(const char* mask_file);

    void mask(T *h_data,
              unsigned int num_channels,
              unsigned int num_samples,
              std::vector<iqrm_omp::WindowMask> &win_masks);

    const std::vector<int>& get_bad_channels() const { return bad_channels_; }

private:
    std::vector<int> bad_channels_;
};

// 显式实例化声明（由 .cpp 定义）
extern template class RfiMarkerCPU<uint8_t>;
extern template class RfiMarkerCPU<uint16_t>;
extern template class RfiMarkerCPU<uint32_t>;

#endif // _RFIMARKER_CPU_H
