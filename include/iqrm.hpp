#pragma once
#ifndef _IQRM_HPP
#define _IQRM_HPP

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <omp.h>

#include "rfi.h"


namespace iqrm_omp {


struct Config {
    unsigned int nchan_total = 0;   // 必填：总通道数 NCHAN（用于 data[t*NCHAN + ch] & 输出长度）
    float  radius_frac  = 0.10f;    // 半径占子带宽度比例
    float  nsigma       = 3.0f;     // σ 阈值
    double geofactor    = 1.5;      // lag 的几何步进
    double win_sec      = 5.0;
    double hop_sec      = 2.5;
    bool   include_tail = true;
};


inline Config& cfg(){ static Config c; return c; }
inline void set_total_channels(unsigned int nchan_total){ cfg().nchan_total = nchan_total; }
inline void set_iqrm_params(float radius_frac=0.10f, float nsigma=3.0f, double geofactor=1.5){
    cfg().radius_frac = std::max(0.0f, radius_frac);
    cfg().nsigma      = std::max(0.0f, nsigma);
    cfg().geofactor   = geofactor > 1.0 ? geofactor : 1.5;
}
inline void set_window_seconds(double win_sec, double hop_sec, bool include_tail=true){
    cfg().win_sec      = std::max(0.0, win_sec);
    cfg().hop_sec      = std::max(0.0, hop_sec);
    cfg().include_tail = include_tail;
}

//================ 工具：lags / 分位数 / Tukey =================
static inline std::vector<int> gen_lags(int radius, double geofactor){
    std::vector<int> lags;
    if (radius <= 0) return lags;
    int lag = 1;
    while (lag <= radius){
        lags.push_back(+lag);
        lags.push_back(-lag);
        int nxt = static_cast<int>(geofactor * lag);
        lag = std::max(nxt, lag + 1);
    }
    return lags;
}
static inline float percentile_sorted(const std::vector<float>& v, size_t m, float q){
    if (m==0) return std::numeric_limits<float>::quiet_NaN();
    if (m==1) return v[0];
    float pos = q * float(m-1);
    size_t lo = (size_t)std::floor(pos), hi = (size_t)std::ceil(pos);
    float w = pos - float(lo);
    return v[lo] + (v[hi]-v[lo]) * w;
}
static inline bool tukey_on_vector(const std::vector<float>& d, float& med, float& std_est){
    std::vector<float> vals; vals.reserve(d.size());
    for (float u: d) if (std::isfinite(u)) vals.push_back(u);
    if (vals.empty()) return false;
    std::sort(vals.begin(), vals.end());
    float q1 = percentile_sorted(vals, vals.size(), 0.25f);
    med      = percentile_sorted(vals, vals.size(), 0.50f);
    float q3 = percentile_sorted(vals, vals.size(), 0.75f);
    std_est  = std::abs((q3 - q1) / 1.349f);
    return (std_est > 0.f) && std::isfinite(std_est);
}
// 参考 iqrm_apollo 的“常数差填充”边界
static inline void lagged_diff_with_clipping(const std::vector<float>& x, int lag,
                                             std::vector<float>& d)
{
    const size_t N = x.size();
    d.resize(N);
    if (lag == 0){ std::fill(d.begin(), d.end(), 0.0f); return; }
    if (lag < 0){
        const size_t cut = N + lag;            // N - |lag|
        const float edge = x[cut] - x[N-1];
        #pragma omp parallel for if (N >= 2048)
        for (int k=0; k<(int)N; ++k){
            if ((size_t)k < cut) d[k] = x[k] - x[k - lag]; // k + |lag|
            else                 d[k] = edge;
        }
    } else {
        const float edge = x[lag] - x[0];
        #pragma omp parallel for if (N >= 2048)
        for (int k=0; k<(int)N; ++k){
            if ((size_t)k < (size_t)lag) d[k] = edge;
            else                         d[k] = x[k] - x[k - lag];
        }
    }
}
// 对通道统计量 x（长度=Csub）做 IQRM，返回区间内的 0/1 mask（长度=Csub）
static inline void iqrm_on_stat(const std::vector<float>& x,
                                float radius_frac, float nsigma, double geofactor,
                                std::vector<uint8_t>& mask_out)
{
    const unsigned C = (unsigned)x.size();
    mask_out.assign(C, 0u);
    if (C == 0) return;
    const int radius = std::max(1, (int)std::floor(radius_frac * (float)C));
    auto lags = gen_lags(radius, geofactor);

    std::vector<int> E_i; E_i.reserve(C * lags.size() / 16 + 16);
    std::vector<int> E_j; E_j.reserve(C * lags.size() / 16 + 16);
    std::vector<float> d; d.reserve(C);
    std::vector<uint8_t> flag(C);

    for (int lag: lags){
        lagged_diff_with_clipping(x, lag, d);
        float med, s;
        if (!tukey_on_vector(d, med, s)){ std::fill(flag.begin(), flag.end(), 0u); }
        else {
            const float thr = nsigma * s;
            #pragma omp parallel for
            for (int i=0; i<(int)C; ++i) flag[i] = (d[i] - med) > thr ? 1u : 0u;
        }
        for (unsigned k=0; k<C; ++k){
            if (!flag[k]) continue;
            int j;
            if (lag < 0){
                if ((int)k < (int)C + lag) j = k - lag;
                else                        j = (int)C - 1;
            } else {
                if ((int)k < lag)           j = 0;
                else                        j = k - lag;
            }
            E_i.push_back((int)k);
            E_j.push_back(j);
        }
    }
    if (E_i.empty()) return;

    std::vector<int> recv(C,0), cast_(C,0);
    #pragma omp parallel for
    for (int k=0; k<(int)E_i.size(); ++k){
        int i = E_i[k], j = E_j[k];
        #ifdef _OPENMP
        #pragma omp atomic
        #endif
        recv[i] += 1;
        #ifdef _OPENMP
        #pragma omp atomic
        #endif
        cast_[j] += 1;
    }
    for (size_t k=0; k<E_i.size(); ++k){
        int i = E_i[k], j = E_j[k];
        if (cast_[j] < recv[i]) mask_out[i] = 1u;
    }
}

//================ 每窗输出（长度=NCHAN 的坏道表） ================
struct WindowMask {
    unsigned t0;                 // 本窗起始 sample（含）
    unsigned t1;                 // 本窗结束 sample（不含）
    std::vector<uint8_t> mask;   // 长度 = NCHAN；[chan_start,chan_end) 内为 IQRM 结果，其它通道=0
};

//============= 入口：返回“每窗 × NCHAN”的 0/1 坏道表 =============
template <typename T>
inline std::vector<WindowMask>
iqrm(const T* data, unsigned int chan_start, unsigned int chan_end,
     double tsamp, unsigned int nsample, int mode)
{
    static_assert(std::is_unsigned<T>::value && sizeof(T) <= 4,
                  "T must be uint8_t/uint16_t/uint32_t");

    const auto& Cfg = cfg();
    const unsigned int NCHAN = Cfg.nchan_total;
    if (!data || NCHAN==0 || nsample==0 || chan_start>=chan_end || chan_end>NCHAN)
        return {};

    const unsigned Csub = chan_end - chan_start;

    const unsigned W = (Cfg.win_sec > 0.0 ? (unsigned)std::floor(Cfg.win_sec / tsamp) : 0u);
    const unsigned H = (Cfg.hop_sec > 0.0 ? (unsigned)std::floor(Cfg.hop_sec / tsamp) : W);

    // 计算一个窗口 [t0,t1) 的区间统计量 x（长度=Csub），mode=0 Mean，1 Std
    auto compute_stat_in_win = [&](unsigned t0, unsigned t1, std::vector<float>& x){
        x.assign(Csub, 0.f);
        if (mode == 0){ // Mean
            #pragma omp parallel for
            for (int i=0; i<(int)Csub; ++i){
                unsigned ch = chan_start + (unsigned)i;
                double sum = 0.0;
                for (unsigned t=t0; t<t1; ++t)
                    sum += (double)data[(size_t)t * NCHAN + ch];
                x[i] = (float)(sum / double(t1 - t0));
            }
        } else { // Std
            #pragma omp parallel for
            for (int i=0; i<(int)Csub; ++i){
                unsigned ch = chan_start + (unsigned)i;
                double sum = 0.0;
                for (unsigned t=t0; t<t1; ++t)
                    sum += (double)data[(size_t)t * NCHAN + ch];
                double mean = sum / double(t1 - t0);
                double s2 = 0.0;
                for (unsigned t=t0; t<t1; ++t){
                    double v = (double)data[(size_t)t * NCHAN + ch] - mean;
                    s2 += v*v;
                }
                s2 /= double(t1 - t0);
                x[i] = (float)std::sqrt(std::max(0.0, s2));
            }
        }
    };

    std::vector<WindowMask> out;

    auto push_window = [&](unsigned t0, unsigned t1){
        std::vector<float> stat;
        compute_stat_in_win(t0, t1, stat);

        // 区间内 IQRM → 得到 Csub 长度的 0/1 mask
        std::vector<uint8_t> submask;
        iqrm_on_stat(stat, Cfg.radius_frac, Cfg.nsigma, Cfg.geofactor, submask);

        // 扩展成 NCHAN 长度：区间外=0，区间内拷贝 submask
        std::vector<uint8_t> full(NCHAN, 0u);
        if (!submask.empty()){
            std::copy(submask.begin(), submask.end(), full.begin() + chan_start);
        }
        out.push_back(WindowMask{t0, t1, std::move(full)});
    };

    if (W == 0 || W >= nsample){
        // 整段一个窗
        push_window(0, nsample);
    } else {
        // 计算所有窗口的起止点
        std::vector<std::pair<unsigned, unsigned>> win_ranges;
        unsigned t0 = 0;
        while (t0 + W <= nsample){
            win_ranges.emplace_back(t0, t0 + W);
            t0 += (H ? H : W);
        }
        if (Cfg.include_tail && t0 < nsample){
            unsigned t1 = nsample;
            if (t1 > t0 + 4) win_ranges.emplace_back(t0, t1); // 极短尾巴可跳过
        }
        // OpenMP 并行处理每个窗口
        std::vector<WindowMask> win_out(win_ranges.size());
        #pragma omp parallel for if(win_ranges.size() > 1)
        for (int i = 0; i < (int)win_ranges.size(); ++i) {
            std::vector<WindowMask> tmp;
            auto [t0, t1] = win_ranges[i];
            // push_window 逻辑内联
            std::vector<float> stat;
            compute_stat_in_win(t0, t1, stat);
            std::vector<uint8_t> submask;
            iqrm_on_stat(stat, Cfg.radius_frac, Cfg.nsigma, Cfg.geofactor, submask);
            std::vector<uint8_t> full(NCHAN, 0u);
            if (!submask.empty()){
                std::copy(submask.begin(), submask.end(), full.begin() + chan_start);
            }
            win_out[i] = WindowMask{t0, t1, std::move(full)};
        }
        // 合并结果
        for (auto& w : win_out) out.push_back(std::move(w));
    }
    return out;
}

} // namespace iqrm_omp

template <typename T>
std::vector<iqrm_omp::WindowMask> rfi_iqrm(T* input, size_t chan_start, size_t chan_end, size_t ndata, size_t nchans, double tsamp, rficonfig rficfg) {
  if (!rficfg.use_iqrm) return {};
  iqrm_omp::set_total_channels(nchans);
  iqrm_omp::set_iqrm_params(rficfg.iqrm_cfg.radius_frac,
                           rficfg.iqrm_cfg.nsigma,
                           rficfg.iqrm_cfg.geofactor);
  iqrm_omp::set_window_seconds(rficfg.iqrm_cfg.win_sec,
                               rficfg.iqrm_cfg.hop_sec,
                               rficfg.iqrm_cfg.include_tail);                           
  return iqrm_omp::iqrm(input, chan_start, chan_end, tsamp, ndata, rficfg.iqrm_cfg.mode);                               
}
#endif // _IQRM_HPP