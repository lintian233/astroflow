import os, random
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt   # 仅为调试可删
from astroflow.io.filterbank import Filterbank, SpectrumType
from astroflow.io.psrfits import PsrFits
from astroflow.dedispered import dedisperse_spec
from astroflow.dataset.generate import get_ref_freq_toa

# ---------------- 基本常量 ----------------
DISPERSION_CONSTANT = 4148.808  # MHz² pc⁻¹ cm³ ms
BASE_OUT = '/home/lingh/work/astroflow/ql/simulated_candidates'

FRB_IMG_DIR   = f'{BASE_OUT}/frb/images'
FRB_LAB_DIR   = f'{BASE_OUT}/frb/labels'
RFI_IMG_DIR   = f'{BASE_OUT}/rfi/images'
RFI_LAB_DIR   = f'{BASE_OUT}/rfi/labels'
WEAK_FRB_IMG_DIR = f"{BASE_OUT}/weak_frb/images"
WEAK_FRB_LAB_DIR = f"{BASE_OUT}/weak_frb/labels"
for d in [FRB_IMG_DIR, FRB_LAB_DIR, RFI_IMG_DIR, RFI_LAB_DIR, WEAK_FRB_IMG_DIR, WEAK_FRB_LAB_DIR]:
    os.makedirs(d, exist_ok=True)

# 每个候选要生成的背景图片张数
BG_NUM = 1  # ← 改这个即可

# ---------------- 基础工具 ----------------
def dm_to_delay_samples(dm, f_low, f_high, dt_sample):
    delay_s = DISPERSION_CONSTANT * dm * (1.0/f_low**2 - 1.0/f_high**2)
    return int(round(delay_s / dt_sample))

def delay_samples_to_dm(delay_samples, f_low, f_high, dt_sample):
    delay_s = delay_samples * dt_sample
    return delay_s / (DISPERSION_CONSTANT * (1.0/f_low**2 - 1.0/f_high**2))


def generate_test_spectrogram(file_path, dm, toa, pulse_width_ms, pulse_amp_ratio,
                              t_len, freq_min, freq_max):

    base = Filterbank(file_path) if file_path.endswith('.fil') else PsrFits(file_path)
    hdr = base.core_data[1]
    dt      = hdr.tsamp
    n_f     = hdr.nchans
    n_t     = hdr.ndata
    f_low   = hdr.fch1
    f_off   = hdr.foff
    f_high  = f_low + f_off * (n_f - 1)

    toa_samples   = int(toa / dt)
    width_samples = max(1, int(pulse_width_ms / 1000.0 / dt))  # 避免 width=0
    base.settype(SpectrumType.CUSTOM)                          # type: ignore

    # 取谱并转型一次，后续就地更新
    spec = base.get_spectrum().astype(np.uint16, copy=False)   # type: ignore
    max_amp = int(spec.max())
    amp = pulse_amp_ratio * max_amp
    noise_scale = 0.005 * amp

    # -------- 频率与延时（向量化） --------
    # 用等差生成频率（更快且与数据一致），并筛选频段
    idx_all = np.arange(n_f, dtype=np.int32)
    freqs   = f_low + f_off * idx_all
    mask    = (freqs >= freq_min) & (freqs <= freq_max)
    valid_idx = idx_all[mask]
    if valid_idx.size > 0:
        inv_fhigh2 = 1.0 / (f_high * f_high)
        inv_f2     = 1.0 / (freqs[valid_idx] * freqs[valid_idx])
        # 延时（秒） -> 样本数（四舍五入）
        delays = np.rint(DISPERSION_CONSTANT * dm * (inv_f2 - inv_fhigh2) / dt).astype(np.int64)

        for j in range(valid_idx.size):
            i = int(valid_idx[j])
            center = toa_samples + int(delays[j]) + np.random.randint(-15, 15)  # TOA添加随机偏移
            if center <= 0 or center >= n_t:
                continue

            j0 = np.random.randint(-30, 30)
            j1 = np.random.randint(-30, 30)

            t0 = max(0, center - width_samples) + j0
            t1 = min(n_t, center + width_samples) + j1
            if t1 <= t0:
                continue

            t_idx = np.arange(t0, t1, dtype=np.int32)

            g = amp * np.exp(-0.5 * ((t_idx - center) / float(width_samples))**2, dtype=np.float64)
            if noise_scale > 0:
                g += np.random.normal(0.0, noise_scale, size=g.shape)

            g = g.astype(np.uint16, copy=False)
            spec[t0:t1, i] += g

    
    np.clip(spec, 0, 255, out=spec)
    spec = spec.astype(np.uint8, copy=False)
    base.spectrumset(spec.ravel())                                # type: ignore

    # 截取 pulse 附近 t_len
    t_clip  = int(t_len / dt)
    start_t = max(0, toa_samples)
    end_t   = start_t + t_clip
    if end_t > n_t:
        end_t  = n_t
        start_t = max(0, end_t - t_clip)

    clip_spec = spec[start_t:end_t]
    return clip_spec, base

# ---------- DMT 相关 ----------
def dedisperse_get_list(spectrum, dm_low, dm_high, f_start, f_end,
                        dm_step, t_down, t_sample):
    """一次性返回全部 DmTime 对象"""
    return dedisperse_spec(spectrum, dm_low, dm_high,
                           f_start, f_end, dm_step,
                           t_down, t_sample,
                           maskfile="/home/lingh/work/astroflow/python/none.txt")

def split_dmt_by_toa(dmts, toa, min_gap=0.0):
    """
    返回:
      pulse_dmt: 使 (tstart < toa < tend) 的 DmTime
      bg_pool:   满足 |(tstart+tend)/2 - toa| >= min_gap 的 DmTime 列表
    """
    pulse = None
    bg_pool = []
    for d in dmts:
        mid = 0.5 * (d.tstart + d.tend)
        if d.tstart < toa < d.tend:
            pulse = d
        elif abs(mid - toa) >= min_gap:
            bg_pool.append(d)
    return pulse, bg_pool

def save_dmt_image(dmt, out_path, vmin_pct=0, vmax_pct=100):
    # img = np.array(dmt.data, dtype=np.float32)          # type: ignore
    # vmin, vmax = np.percentile(img, vmin_pct), np.percentile(img, vmax_pct)
    # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    img = dmt.data
    cv2.imwrite(out_path, img)

def gen_label(dm, toa, imgsize, dm_low, dm_high, t_start, t_end):
    dm_range = dm_high - dm_low
    toa_range = t_end - t_start
    dm_pos = int((dm - dm_low) / dm_range * imgsize[0])
    toa_pos = int((toa - t_start) / toa_range * imgsize[1])
    dm_pos = min(max(dm_pos, 0), imgsize[0] - 1)
    toa_pos = min(max(toa_pos, 0), imgsize[1] - 1)
    dm_pos = np.round(dm_pos / imgsize[0], 2)
    toa_pos = np.round(toa_pos / imgsize[1], 2)
    return (0, toa_pos, dm_pos, 0.2, 0.2)

# ---------------- 主流程 ----------------
def main():
    # ---------- 超参数 ----------
    CAND_NUM   = 2000
    file_dir   = '/data/QL/lingh/FAST_RFI'
    fil_files  = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if (f.endswith('.fil') or f.endswith('.fits'))]
    
    toa_range  = (1, 4)       # s
    # 候选随机参数空间
    cand_dm_rng   = (400, 670)
    freq_min_rng  = (1000, 1300)
    freq_max_rng  = (1150, 1470)     # 确保 freq_max - freq_min ≥100
    width_rng_ms  = (0.1, 6)
    amp_ratio_rng = (0.001, 0.02)
    t_clip_len    = 0.5              # s

    # dedispersion 参数
    dm_low, dm_high, dm_step   = 350, 750, 0.6
    f_start, f_end             = 1025.0, 1450.0
    t_down, t_sample           = 1, 0.5

    for i in tqdm(range(CAND_NUM), desc='Generating'):
        # ---- 随机化参数 ----
        file_path  = random.choice(fil_files)
        dm         = np.random.uniform(*cand_dm_rng)
        toa        = np.random.uniform(*toa_range)
        width_ms   = np.random.uniform(*width_rng_ms)
        amp_ratio  = np.random.uniform(*amp_ratio_rng)
        f_min      = np.random.uniform(*freq_min_rng)
        f_max      = max(f_min+100, np.random.uniform(*freq_max_rng))

        # ---- 生成含脉冲的滤波器数据 ----
        clip_spec, base = generate_test_spectrogram(
            file_path, dm, toa, width_ms, amp_ratio,
            t_clip_len, f_min, f_max)

        ref_toa = get_ref_freq_toa(base.header(), f_end, toa, dm)  # pulse 参考时标

        dmts = dedisperse_get_list(base, dm_low, dm_high, f_start, f_end,
                                   dm_step, t_down, t_sample)
        frb_dmt, bg_pool = split_dmt_by_toa(dmts, ref_toa, min_gap=1)

        if frb_dmt is None:
            print(f'[Skip] 未找到包含脉冲的 DMT (DM={dm:.2f}, TOA={toa:.2f})')
            continue

        # ---- 保存 FRB 样本 ----
        stem = os.path.splitext(os.path.basename(file_path))[0]
        ftag = (f'{stem}_dm_{dm:.2f}_toa_{toa:.2f}_pw_{width_ms:.2f}_'
                f'pa_{amp_ratio:.2f}_freq_{f_min:.2f}_{f_max:.2f}')
        

        if (width_ms <= 0.6) or (width_ms < 1 and amp_ratio <= 0.06):
            # 保存弱 FRB 样本
            weak_frb_img = f'{WEAK_FRB_IMG_DIR}/{ftag}.png'
            weak_frb_lab = f'{WEAK_FRB_LAB_DIR}/{ftag}.txt'
            save_dmt_image(frb_dmt, weak_frb_img, 0, 100)
            lab = gen_label(dm, ref_toa, (512,512), dm_low, dm_high,
                            frb_dmt.tstart, frb_dmt.tend)
            with open(weak_frb_lab, 'w') as f:
                f.write(' '.join(map(str, lab)) + '\n')

            continue  # 跳过 FRB 样本保存

        frb_img = f'{FRB_IMG_DIR}/{ftag}.png'
        frb_lab = f'{FRB_LAB_DIR}/{ftag}.txt'
        save_dmt_image(frb_dmt, frb_img, 0, 100)
        lab = gen_label(dm, ref_toa, (512,512), dm_low, dm_high,
                        frb_dmt.tstart, frb_dmt.tend)
        with open(frb_lab, 'w') as f:
            f.write(' '.join(map(str, lab)) + '\n')

        # ---- 保存背景样本 (空标签)，最多 BG_NUM 张 ----
        if bg_pool:
            # 随机无放回抽取
            pick_n = min(BG_NUM, len(bg_pool))
            idxs = np.random.choice(len(bg_pool), size=pick_n, replace=False)
            for kk, idx in enumerate(idxs):
                bg_dmt = bg_pool[idx]
                bg_tag  = f'bg_{stem}_{i:06d}_b{kk:02d}'
                bg_img  = f'{RFI_IMG_DIR}/{bg_tag}_{dm:.2f}_{toa:.2f}.png'
                bg_lab  = f'{RFI_LAB_DIR}/{bg_tag}_{dm:.2f}_{toa:.2f}.txt'
                save_dmt_image(bg_dmt, bg_img, 0, 100)
                open(bg_lab, 'w').close()   # 空标签文件

        print(f'[OK] {i}/{CAND_NUM}  FRB→{os.path.basename(frb_img)} '
              f' BG×{min(BG_NUM, len(bg_pool)) if bg_pool else 0}')

if __name__ == '__main__':
    main()
