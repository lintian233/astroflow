#ifndef _RFIMARKER_H
#define _RFIMARKER_H

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cstdint>
#include <iqrm.hpp>


/**
 * GPU RFI marker:
 * - load_mask(): 在 CPU 读入坏道列表，并上传到 GPU
 * - mark_rfi(): 传入 device 指针 (d_data)，在 kernel 中把坏道通道清零
 *
 * 数据布局假设为: data[sample * num_channels + chan]  (row-major: time-major)
 */
template <typename T>
class RfiMarker {
public:
    RfiMarker();
    explicit RfiMarker(const char* mask_file);
    explicit RfiMarker(const std::string& mask_file) : RfiMarker(mask_file.c_str()) {}
    ~RfiMarker();

    // 将坏道置零（在 GPU 上执行）。d_data 必须是 device 指针
    void mark_rfi(T* d_data,
                  unsigned int num_channels,
                  unsigned int num_samples,
                  cudaStream_t stream = 0);

    void zero_dm_filter(T* d_data,
                        unsigned int num_channels,
                        unsigned int num_samples,
                        float alpha = 1.0f,
                        bool clip_to_zero = true,
                        cudaStream_t stream = 0);

    void zero_dm_filter(T* d_data,
                                unsigned int num_channels,
                                unsigned int num_samples,
                                unsigned int chan_start,
                                unsigned int chan_end,
                                float alpha = 1.0f,
                                bool clip_to_zero = true,
                                cudaStream_t stream = 0);

    // 重新加载掩码文件（会同步上传到 GPU）；文件不存在或为空则视为无坏道
    void load_mask(const char* mask_file);

    void mask(T* d_data,
              unsigned int num_channels,
              unsigned int num_samples,
              const std::vector<iqrm_omp::WindowMask>& win_masks,
              cudaStream_t stream = 0);

    // Host 侧只读坏道列表
    const std::vector<int>& get_bad_channels() const { return bad_channels_; }

private:
    void upload_bad_channels_to_device();

    std::vector<int> bad_channels_;  // host: 坏道索引
    int* d_bad_channels_ = nullptr;  // device: 坏道索引
    size_t n_bad_ = 0;
};

// ------------ 显式实例化声明（由 .cu 文件提供定义） ------------
extern template class RfiMarker<uint8_t>;
extern template class RfiMarker<uint16_t>;
extern template class RfiMarker<uint32_t>;

#endif // _RFIMARKER_H
