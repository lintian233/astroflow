# AstroFlow 合成数据集生成器

这个模块提供了用于生成合成 FRB（快速射电暴）数据集的工具，用于训练机器学习模型。

## 特性

- 生成合成 FRB 候选和背景 RFI 样本
- 支持多种参数配置
- 自动生成 YOLO 格式的标签
- 可区分普通 FRB 和弱 FRB 样本
- 支持命令行和编程式接口
- 支持配置文件

## 安装和设置

确保你已经安装了 AstroFlow 库的所有依赖项。新的合成数据生成功能已经集成到 `astroflow.dataset` 模块中。

## 使用方法

### 1. 命令行界面

最简单的使用方法：

```bash
python scripts/generate_synthetic_dataset.py /path/to/input/data /path/to/output/directory
```

使用配置文件：

```bash
python scripts/generate_synthetic_dataset.py /path/to/input/data /path/to/output/directory --config configs/synthetic_dataset_config.json
```

详细参数配置：

```bash
python scripts/generate_synthetic_dataset.py \\
    /data/QL/lingh/FAST_RFI \\
    /home/lingh/work/astroflow/ql/simulated_candidates \\
    --num-candidates 400 \\
    --bg-samples 2 \\
    --dm-range 320 570 \\
    --width-range 2 6 \\
    --verbose
```

### 2. 编程式接口

```python
from astroflow.dataset.simulator import SimulationConfig, generate_synthetic_dataset

# 创建配置
config = SimulationConfig(
    input_dir="/path/to/input/data",
    output_dir="/path/to/output",
    num_candidates=100,
    dm_range=(320, 570),
    width_range_ms=(2, 6),
)

# 生成数据集
stats = generate_synthetic_dataset(config)
print(f"Generated {stats['frb_samples']} FRB samples")
```

### 3. 使用示例脚本

```bash
python examples/generate_synthetic_example.py
```

## 输出结构

生成的数据集将按以下结构组织：

```
output_directory/
├── frb/
│   ├── images/          # FRB 样本图像
│   └── labels/          # FRB 样本标签（YOLO 格式）
├── weak_frb/
│   ├── images/          # 弱 FRB 样本图像
│   └── labels/          # 弱 FRB 样本标签
└── rfi/
    ├── images/          # 背景/RFI 样本图像
    └── labels/          # 背景样本标签（空文件）
```

## 配置参数

### 主要参数

- `input_dir`: 输入数据目录（包含 .fil 或 .fits 文件）
- `output_dir`: 输出目录
- `num_candidates`: 要生成的候选数量
- `bg_samples_per_candidate`: 每个候选生成的背景样本数

### 信号参数

- `dm_range`: 色散测度范围 (pc cm^-3)
- `toa_range`: 到达时间范围 (秒)
- `width_range_ms`: 脉冲宽度范围 (毫秒)
- `amp_ratio_range`: 振幅比率范围
- `freq_min_range`, `freq_max_range`: 频率范围 (MHz)

### 去色散参数

- `dm_low`, `dm_high`, `dm_step`: 去色散搜索范围和步长
- `f_start`, `f_end`: 搜索频率范围
- `t_down`: 时间下采样因子

### 弱 FRB 判断标准

- `weak_frb_width_threshold`: 宽度阈值 (ms)
- `weak_frb_amp_threshold`: 振幅阈值
- `weak_frb_combined_threshold`: 组合阈值 (ms)

## 命令行选项

使用 `--help` 查看所有可用选项：

```bash
python scripts/generate_synthetic_dataset.py --help
```

主要选项包括：

- `--num-candidates, -n`: 候选数量
- `--bg-samples, -b`: 背景样本数
- `--dm-range`: DM 范围
- `--width-range`: 脉冲宽度范围
- `--config, -c`: 配置文件路径
- `--save-config`: 保存配置到文件
- `--verbose, -v`: 详细输出

## 配置文件格式

配置文件使用 JSON 格式。参见 `configs/synthetic_dataset_config.json` 作为示例。

## 注意事项

1. 确保输入目录包含有效的 .fil 或 .fits 文件
2. 输出目录会自动创建如果不存在
3. 生成过程可能需要一些时间，取决于候选数量和输入数据大小
4. 使用 `--verbose` 选项可以看到详细的进度信息

## 错误处理

- 如果找不到输入文件，脚本会报错并退出
- 如果单个文件处理失败，会跳过该文件并继续处理其他文件
- 所有错误都会记录在输出中

## 示例输出

```
Generating synthetic data: 100%|██████████| 400/400 [00:15<00:00, 26.34it/s]

Generation completed!
Total generated: 395
FRB samples: 298
Weak FRB samples: 97
Background samples: 790
Skipped: 5
Output directory: /path/to/output
```
