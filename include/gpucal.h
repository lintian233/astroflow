/**
 * Project astroflow
 */
#pragma once
#ifndef _GPUCAL_H
#define _GPUCAL_H

#include "data.h"
#include "rfi.h"
#include "filterbank.h"

namespace gpucal {

#define REF_FREQ_END 1
#define REF_FREQ_START 0

template <typename T>
dedisperseddata_uint8
dedispered_fil_cuda(Filterbank &fil, float dm_low, float dm_high,
                    float freq_start, float freq_end, float dm_step,
                    int ref_freq, int time_downsample, 
                    float t_sample, int target_id, std::string mask_file, rficonfig rficfg);

template <typename T>
dedisperseddata_uint8 dedisperse_spec(T *data, Header header, float dm_low,
                                float dm_high, float freq_start, float freq_end,
                                float dm_step, int ref_freq,
                                int time_downsample, float t_sample, int target_id,
                                std::string mask_file, rficonfig rficfg);


template <typename T>
Spectrum<T> dedisperse_spec_with_dm(T *spec, Header header, float dm,
                                    float tstart, float tend,
                                    float freq_start, float freq_end,
                                    std::string maskfile, rficonfig rficfg);

template <typename T>
Spectrum<T> dedispered_fil_with_dm(Filterbank *fil, float tstart, float tend,
                                   float dm, float freq_start, float freq_end,
                                   std::string maskfile, rficonfig rficfg);

/**
 * @brief dedisperse_spec_with_dm 的GPU优化版本
 *
 * 该函数在GPU上执行单DM值的解色散。相比CPU版本：
 * - 利用GPU并行处理多个频道的延迟计算和数据读取
 * - 支持GPU加速的RFI标记（IQRM、静态掩膜）
 * - 更高的内存带宽和计算能力
 *
 * @tparam T 数据类型（uint8_t, uint16_t, uint32_t, float, double）
 * @param spec 输入光谱数据指针（主机内存，行主序：时间×频道）
 * @param header 光谱文件头信息
 * @param dm 色散量（单个值）
 * @param tstart 起始时间（秒）
 * @param tend 结束时间（秒）
 * @param freq_start 起始频率（MHz）
 * @param freq_end 结束频率（MHz）
 * @param maskfile 静态RFI掩膜文件路径
 * @param rficfg RFI配置参数
 * @return Spectrum<T> 解色散后的光谱结构
 *
 * @throws std::invalid_argument 若参数不合法
 */
template <typename T>
Spectrum<T> dedisperse_spec_with_dm_gpu(
    T* spec, Header header, float dm,
    float tstart, float tend,
    float freq_start, float freq_end,
    std::string maskfile, rficonfig rficfg);
                                   
} // namespace gpucal
#endif //_GPUCAL_H
