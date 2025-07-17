import cupy as cp
import numpy as np
import time
import cv2  # 用于图像处理

# 色散常数 (MHz² pc⁻¹ cm³ ms)
DISPERSION_CONSTANT = 4148.808

def dm_to_delay_samples(dm, f_low, f_high, dt_sample):
    """
    将DM值转换为时延样本数
    
    参数:
    - dm: 色散测量值 (pc cm⁻³)
    - f_low: 低频 (MHz)
    - f_high: 高频 (MHz) 
    - dt_sample: 时间采样间隔 (ms)
    
    返回:
    - delay_samples: 时延样本数
    """
    # 色散延迟公式: Δt = K * DM * (1/f_low² - 1/f_high²)
    delay_ms = DISPERSION_CONSTANT * dm * (1.0/(f_low**2) - 1.0/(f_high**2))
    delay_samples = int(cp.round(delay_ms / dt_sample))
    return delay_samples


def delay_samples_to_dm(delay_samples, f_low, f_high, dt_sample):
    """
    将时延样本数转换为DM值
    
    参数:
    - delay_samples: 时延样本数
    - f_low: 低频 (MHz)
    - f_high: 高频 (MHz)
    - dt_sample: 时间采样间隔 (ms)
    
    返回:
    - dm: 色散测量值 (pc cm⁻³)
    """
    delay_ms = delay_samples * dt_sample
    dm = delay_ms / (DISPERSION_CONSTANT * (1.0/(f_low**2) - 1.0/(f_high**2)))
    return dm


def calculate_maxdt(dm_low, dm_high, f_min, f_max, dt_sample):
    """
    根据DM范围计算最大时延样本数
    
    参数:
    - dm_low: 最小DM值 (pc cm⁻³)
    - dm_high: 最大DM值 (pc cm⁻³)
    - f_min: 最小频率 (MHz)
    - f_max: 最大频率 (MHz)
    - dt_sample: 时间采样间隔 (ms)
    
    返回:
    - maxDT: 最大时延样本数
    """
    # 计算最大DM对应的时延
    max_delay_samples = dm_to_delay_samples(dm_high, f_min, f_max, dt_sample)
    return max_delay_samples


def FDMT_GPU(Image, f_min, f_max, dm_low, dm_high, dt_sample, dataType='float32', Verbose=True):
    """
    FDMT主函数 - GPU版本，基于DM范围的实现
    
    参数:
    - Image: 输入功率矩阵 I(f,t), 形状为 [频率, 时间] (可以是numpy或cupy数组)
    - f_min, f_max: 频带的最小和最大频率 (MHz)
    - dm_low, dm_high: DM搜索范围 (pc cm⁻³)
    - dt_sample: 时间采样间隔 (ms)
    - dataType: 数据类型
    - Verbose: 是否输出详细信息
    
    返回:
    - DMT: 色散测量变换结果，形状为 [maxDT+1, 时间]
    - dm_trials: 对应的DM值数组
    """
    # 确保数据在GPU上
    if isinstance(Image, np.ndarray):
        Image = cp.asarray(Image)
    
    F, T = Image.shape
    # 检查输入维度必须是2的幂
    f = int(np.log2(F))
    if F != 2**f:
        raise ValueError(f"频率通道数必须是2的幂，当前为 {F}, 预期:{2**f}")

    # 计算最大时延
    maxDT = calculate_maxdt(dm_low, dm_high, f_min, f_max, dt_sample)
    
    if Verbose:
        print(f"FDMT GPU参数:")
        print(f"  频率范围: {f_min:.1f} - {f_max:.1f} MHz")
        print(f"  频率通道数: {F}")
        print(f"  时间样本数: {T}")
        print(f"  DM范围: {dm_low:.3f} - {dm_high:.3f} pc cm⁻³")
        print(f"  时间采样: {dt_sample:.6f} ms")
        print(f"  最大时延: {maxDT} samples")
    
    # 初始化
    start_time = time.time()
    State = FDMT_initialization_GPU(Image, f_min, f_max, maxDT, dataType)
    if Verbose:
        print("GPU初始化完成")
    
    # 迭代处理
    for i_t in range(1, f + 1):
        State = FDMT_iteration_GPU(State, maxDT, F, f_min, f_max, i_t, dataType, Verbose)
        if Verbose:
            print(f"GPU迭代 {i_t}/{f} 完成")
    
    if Verbose:
        print(f'GPU总耗时: {time.time() - start_time:.3f} 秒')
    
    # 重塑输出
    F_final, dT, T_final = State.shape
    DMT = cp.reshape(State, [dT, T_final])
    
    # 创建DM试验数组
    dm_trials = cp.array([delay_samples_to_dm(i, f_min, f_max, dt_sample) 
                         for i in range(dT)])
    
    # 裁剪到指定的DM范围
    valid_mask = (dm_trials >= dm_low) & (dm_trials <= dm_high)
    DMT = DMT[valid_mask, :]
    dm_trials = dm_trials[valid_mask]
    
    return DMT, dm_trials


def FDMT_initialization_GPU(Image, f_min, f_max, maxDT, dataType):
    """
    FDMT初始化函数 - GPU版本
    """
    F, T = Image.shape
    
    # 计算频率分辨率
    deltaF = (f_max - f_min) / float(F)
    
    # 计算初始最大时延
    deltaT = int(cp.ceil((maxDT - 1) * (1./f_min**2 - 1./(f_min + deltaF)**2) / 
                         (1./f_min**2 - 1./f_max**2)))
    
    # 创建输出数组 - 在GPU上
    Output = cp.zeros([F, deltaT + 1, T], dtype=dataType)
    
    # 初始化第0层（原始数据）
    Output[:, 0, :] = Image
    
    # 计算累积和，为迭代做准备 - 使用GPU并行计算
    for i_dT in range(1, deltaT + 1):
        if i_dT < T:
            Output[:, i_dT, i_dT:] = Output[:, i_dT - 1, i_dT:] + Image[:, :-i_dT]
    
    return Output


def FDMT_iteration_GPU(Input, maxDT, F, f_min, f_max, iteration_num, dataType, Verbose=False):
    """
    FDMT单次迭代函数 - GPU版本
    """
    input_dims = Input.shape
    output_dims = list(input_dims)
    
    # 计算当前迭代的频率带宽
    deltaF = 2**(iteration_num) * (f_max - f_min) / float(F)
    dF = (f_max - f_min) / float(F)
    
    # 计算当前迭代需要的最大deltaT
    deltaT = int(cp.ceil((maxDT - 1) * (1./f_min**2 - 1./(f_min + deltaF)**2) / 
                         (1./f_min**2 - 1./f_max**2)))
    
    if Verbose:
        print(f"  GPU迭代 {iteration_num}: deltaT = {deltaT}")
    
    # 输出维度：频率减半，DM维度更新
    output_dims[0] = output_dims[0] // 2
    output_dims[1] = deltaT + 1
    
    Output = cp.zeros(output_dims, dtype=dataType)
    
    # 偏移参数
    ShiftOutput = 0
    ShiftInput = 0
    T = output_dims[2]
    F_jumps = output_dims[0]
    
    # 频率校正
    correction = dF / 2. if iteration_num > 0 else 0
    
    # 对每个输出频带进行处理
    for i_F in range(F_jumps):
        # 计算频带范围
        f_start = (f_max - f_min) / float(F_jumps) * i_F + f_min
        f_end = (f_max - f_min) / float(F_jumps) * (i_F + 1) + f_min
        f_middle = (f_end - f_start) / 2. + f_start - correction
        f_middle_larger = (f_end - f_start) / 2. + f_start + correction
        
        # 计算当前频带的局部deltaT
        deltaTLocal = int(cp.ceil((maxDT - 1) * (1./f_start**2 - 1./f_end**2) / 
                                  (1./f_min**2 - 1./f_max**2)))
        
        # 对每个DM试验进行处理
        for i_dT in range(deltaTLocal + 1):
            # 计算中间频率对应的DM索引
            dT_middle = round(i_dT * (1./f_middle**2 - 1./f_start**2) / 
                             (1./f_end**2 - 1./f_start**2))
            dT_middle_index = dT_middle + ShiftInput
            
            # 计算较大中间频率对应的DM索引
            dT_middle_larger = round(i_dT * (1./f_middle_larger**2 - 1./f_start**2) / 
                                    (1./f_end**2 - 1./f_start**2))
            
            # 计算剩余DM
            dT_rest = i_dT - dT_middle_larger
            dT_rest_index = dT_rest + ShiftInput
            
            # 时间范围处理
            i_T_min = 0
            i_T_max = dT_middle_larger
            
            # 前半部分：只复制第一个子带的数据
            if i_T_max > i_T_min:
                Output[i_F, i_dT + ShiftOutput, i_T_min:i_T_max] = \
                    Input[2*i_F, dT_middle_index, i_T_min:i_T_max]
            
            # 后半部分：合并两个子带的数据
            i_T_min = dT_middle_larger
            i_T_max = T
            
            if i_T_max > i_T_min and dT_rest_index >= 0 and dT_rest_index < Input.shape[1]:
                shift_end = min(i_T_max - dT_middle_larger, Input.shape[2] - (i_T_min - dT_middle_larger))
                if shift_end > 0:
                    Output[i_F, i_dT + ShiftOutput, i_T_min:i_T_min + shift_end] = \
                        Input[2*i_F, dT_middle_index, i_T_min:i_T_min + shift_end] + \
                        Input[2*i_F + 1, dT_rest_index, 0:shift_end]
    
    return Output


# 测试和使用示例
def FDMT_test_with_dm_range_GPU():
    """
    基于DM范围的FDMT GPU测试
    """
    print("=== 基于DM范围的FDMT GPU测试 ===")
    
    # 观测参数
    f_min = 1000      # MHz - 最低频率
    f_max = 1500      # MHz - 最高频率  
    dm_low = 0.0      # pc cm⁻³ - 最小DM
    dm_high = 100     # pc cm⁻³ - 最大DM
    dt_sample = 4e-5  # 0.04 ms - 时间采样间隔
    
    # 数据参数
    N_f = 2048*2         # 频率通道数（2的幂）
    N_t = 10240      # 时间样本数

    
    print(f"观测参数:")
    print(f"  频率范围: {f_min} - {f_max} MHz")
    print(f"  DM搜索范围: {dm_low} - {dm_high} pc cm⁻³")
    print(f"  时间采样: {dt_sample} s")
    print(f"  总时间: {N_t * dt_sample:.6f} s")
    # 计算最大时延
    maxDT = calculate_maxdt(dm_low, dm_high, f_min, f_max, dt_sample)
    print(f"  计算得到的最大时延: {maxDT} samples, seconds: {maxDT * dt_sample:.6f} s")
    
    # 生成测试数据
    cp.random.seed(42)
    data = cp.random.normal(0, 1, (N_f, N_t)).astype('float32')
    
    # 添加一个已知DM的高斯脉冲
    test_dm = 60  # pc cm⁻³
    pulse_time = 500
    pulse_amplitude = 10.0
    pulse_width = 20  # 高斯脉冲的标准差（样本数）
    
    # 为每个频率通道添加相应延迟的高斯脉冲
    freqs = cp.linspace(f_min, f_max, N_f)
    for i, freq in enumerate(freqs):
        delay_samples = dm_to_delay_samples(test_dm, freq, f_max, dt_sample)
        pulse_center = pulse_time + delay_samples
        
        # 生成高斯脉冲
        if pulse_center < N_t:
            # 定义高斯脉冲的时间范围
            pulse_start = max(0, int(pulse_center - 3 * pulse_width))
            pulse_end = min(N_t, int(pulse_center + 3 * pulse_width))
            
            # 生成高斯分布
            time_indices = cp.arange(pulse_start, pulse_end)
            gaussian_pulse = pulse_amplitude * cp.exp(-0.5 * ((time_indices - pulse_center) / pulse_width) ** 2)
            
            # 添加到数据中
            data[i, pulse_start:pulse_end] += gaussian_pulse
    
    print(f"\n添加测试脉冲:")
    print(f"  DM = {test_dm} pc cm⁻³")
    print(f"  时间位置 = {pulse_time}")
    
    # 执行FDMT GPU
    fdmt_result, dm_trials = FDMT_GPU(data, f_min, f_max, dm_low, dm_high, 
                                      dt_sample, 'float32', Verbose=True)
    
    # 找到峰值
    peak_dm_idx, peak_time_idx = cp.unravel_index(cp.argmax(fdmt_result), fdmt_result.shape)
    detected_dm = dm_trials[peak_dm_idx]
    peak_value = cp.max(fdmt_result)
    
    print(f"\nGPU检测结果:")
    print(f"  检测到的DM: {float(detected_dm):.3f} pc cm⁻³ (真实值: {test_dm})")
    print(f"  时间位置: {int(peak_time_idx)} (输入位置: {pulse_time})")
    print(f"  峰值强度: {float(peak_value):.3f}")
    print(f"  DM误差: {abs(float(detected_dm) - test_dm):.3f} pc cm⁻³")
    
    return fdmt_result, dm_trials, detected_dm, test_dm

def plot_results_gpu(fdmt_result, dm_trials, detected_dm, true_val):
    """
    绘制GPU计算结果
    """
    import matplotlib.pyplot as plt
    
    # 将GPU数据转换为CPU数据用于绘图
    fdmt_result_cpu = cp.asnumpy(fdmt_result) if hasattr(fdmt_result, 'get') else fdmt_result
    dm_trials_cpu = cp.asnumpy(dm_trials) if hasattr(dm_trials, 'get') else dm_trials
    
    fdmt_result_resized = cv2.resize(fdmt_result_cpu, (512, 512))  # 调整图像大小以适应显示
    plt.figure(figsize=(10, 10))
    plt.imshow(fdmt_result_resized, aspect='auto', origin='lower',
               extent=[0, fdmt_result_resized.shape[1], dm_trials_cpu[0], dm_trials_cpu[-1]],
               cmap='viridis')
    plt.xlabel('Time Samples')
    plt.ylabel('DM (pc cm⁻³)')
    plt.title('FDMT GPU Result')
    
    plt.savefig('fdmt_gpu_result.png', dpi=150, bbox_inches='tight')
    print("GPU结果图保存为: fdmt_gpu_result.png")


if __name__ == "__main__":
    # 运行GPU测试
    result, dm_vals, detected, true_val = FDMT_test_with_dm_range_GPU()
    plot_results_gpu(result, dm_vals, detected, true_val)
