# 使用 AstroFlow 处理 FAST PREX FRB 数据集

本教程提供了使用 `astroflow` 流水线处理 FAST PREX 数据集的全面分步指南。我们将涵盖数据采集、流水线配置、执行和结果解读。

该数据集包含已知具有前导辐射的快速射电暴（FRB）的观测数据，使其成为单脉冲和FRB搜索流水线的绝佳验证集。

## 先决条件

- **AstroFlow 安装**: 确保 `astroflow` 已安装并可执行。请参阅[安装指南](./README_zh-CN.md#methods)。
- **磁盘空间**: 数据集需要大约 **200 GB** 的可用磁盘空间。
- **依赖项**: `wget` 和标准的 Unix 命令行工具。

---

## 步骤 1: 下载数据集

该数据集托管在科学数据银行（Science Data Bank）上。

### 1.1. 获取数据 URL

首先，我们需要一个包含数据集中所有文件 URL 的列表。

1.  访问数据集主页：[FAST_PREX Dataset](https://www.scidb.cn/en/detail?dataSetId=3b3cf2f75a74419b89a56cc9626af2a0)
2.  在页面底部，点击 **`Get All URLs`** 按钮。
3.  将完整的 URL 列表保存到名为 `fast_prex.txt` 的本地文本文件中。

### 1.2. 执行批量下载

我们将使用 `xargs` 和 `wget` 来执行并行下载，这可以显著加快下载过程。以下命令可同时下载最多 8 个文件。

```bash
# 为数据创建一个目录
mkdir -p fast_prex_data && cd fast_prex_data

#提取FRB链接，并保存为fast_prex_frbs.txt
cat fast_prex.txt | grep FRB > fast_prex_frbs.txt

# 运行并行下载
xargs -P 8 -n 1 -I {} bash -c '
  url="$1"
  # 从 URL 中提取文件名，处理不同的 URL 格式
  filename=$(printf "%s" "$url" | sed -n "s/.*[?&]fileName=\([^&]*\).*/\1/p")
  filename=${filename:-$(basename "${url%%\?*}")}
  
  # 使用断点续传 (-c) 和清晰的进度条进行下载
  echo "正在下载: $filename"
  wget -c --show-progress --progress=bar:force:noscroll -O "$filename" "$url"
' _ {} < fast_prex_frbs.txt
```

此脚本会遍历 `fast_prex.txt` 中的每个 URL：
- **`-P 8`**: 最多并行运行 8 个 `wget` 进程。
- **`-c`**: 允许 `wget` 恢复中断的下载。
- **`-O "$filename"`**: 使用从 URL 中提取的正确名称保存文件。

下载完成后，请验证 `fast_prex_data/` 目录的内容。

---

## 步骤 2: 准备数据集文件

`dataset` 模式需要一个预定义的候选体列表。此文件包含数据集中已知 FRB 的基准真相信息（文件、DM、TOA）。

```bash
wget -O fast_prex_candidate.csv https://raw.githubusercontent.com/lintian233/astroflow/main/docs/candidates/fast_prex_candidate.csv
```

**重要提示：** 此文件**不用于**指导检测过程。它是在搜索完成后用作基准真相（ground truth）的参考。流水线使用此列表将其找到的候选体与已知的FRB进行交叉匹配，从而能够自动计算召回率（recall）和精确率（precision）等性能指标。检测参数在YAML文件中独立配置。

---

## 步骤 3: 配置搜索流水线

创建一个名为 `fast_prex_search.yaml` 的 YAML 配置文件。该文件用于指示 `astroflow` 如何处理数据。

```yaml
# 保存为 fast_prex_search.yaml

# --- 输入/输出配置 ---
input: path/to/fast_prex_data/      # 包含已下载 FITS 文件的目录
output: path/to/fast_prex_output/   # 用于存放所有流水线输出的目录
mode: dataset                       # `dataset` 模式根据候选体列表处理文件
candpath: /path/to/fast_prex_candidate.csv # 候选体 CSV 文件的路径

# --- 处理与性能 ---
dedgpu: 0                 # 用于色散延迟消除的 GPU 设备 ID
detgpu: 0                 # 用于 AI 检测的 GPU 设备 ID (在多 GPU 环境下可使用不同 ID)
cputhread: 32             # 用于 I/O 和预处理的 CPU 线程数
plotworker: 16            # 用于生成候选体绘图的并行进程数

# --- 检测参数 ---
modelname: yolov11n       # 用于检测的 AI 模型
# modelpath: yolo11n_0816_v1.pt # 可选：自定义训练模型的路径
confidence: 0.4           # 检测置信度阈值 (0.0 到 1.0)
timedownfactor: 8         # 检测前的时间序列降采样因子，可增加对更宽脉冲的灵敏度

# --- RFI 缓解 ---
rfi: 
  use_mask: 0             # (0/1) 设置为 1 以使用外部静态通道掩码
  use_iqrm: 1             # (0/1) 启用 GPU 加速的 IQRM 算法
  use_zero_dm: 0          # (0/1) 启用零 DM 擦除 

iqrm:
  mode: 0                 # 0=均值, 1=标准差。用于异常值检测的统计方法
  radius_frac: 0.1        # 用于自相关函数中延迟选择的半径分数
  nsigma: 7.0             # 标记 RFI 的阈值。值越高越保守
  geofactor: 1.5          # 延迟级数的几何因子
  win_sec: 0              # 窗口大小（秒），0 表示处理全部数据
  hop_sec: 6.04           # 滑动窗口分析的步长（秒）
  include_tail: true      # 处理末尾剩余的数据块

# maskdir: /path/to/FAST_PREFIX_MASK # 可选：如果 use_mask=1，则提供 .bad_chans 文件目录

# --- 搜索空间 ---
tsample:
  - name: t0
    t: 0.5 # 秒。每个被处理数据块的持续时间

dmrange:
  - name: dm100_700
    dm_low: 100
    dm_high: 700
    dm_step: 1 # pc cm^-3。对于高 DM 的 FRB 搜索，步长为 1 是典型值

freqrange:
  - name: Lband
    freq_start: 1000
    freq_end: 1499.5 # MHz。FAST 的 L 波段全覆盖

# --- 绘图配置 ---
dmtconfig:
  minpercentile: 0
  maxpercentile: 99.9

specconfig:
  minpercentile: 0    
  maxpercentile: 100
  tband: 120 # 毫秒。动态谱图的时间窗口
  mode: subband # 'subband' 或 'standard' 或 'detrend'
```

**关键参数解释:**

- **`mode: dataset`**: 此模式旨在处理一系列数据文件。它会对每个文件执行盲搜，然后在搜索完成后使用 `candpath` 文件中的基准信息（DM、时间）来自动对检测到的候选体进行分类（例如，分为 `candidate`、`detect` 或 `background`），并计算性能指标。它非常适合用于流水线的验证、性能测试。
- **`timedownfactor: 8`**: FAST 数据具有很高的时间分辨率。通过因子 8 进行降采样，可以使流水线对时间上更宽的脉冲（这在散射的 FRB 中很常见）更敏感，同时也能减少计算负荷。
- **`iqrm` 设置**: 迭代四分位距缓解 (IQRM) 算法的设置被调整得较为积极 (`nsigma: 7.0`)，以处理 FAST 望远镜经常遇到的复杂 RFI 环境。
- **`dm_step: 1`**: 在高色散量（DM > 100 pc cm⁻³）下，单个通道内的色散弥散效应变得比步长为 1.0 的相邻 DM 试验之间的弥散效应更大。这使其成为一种计算效率高的选择，而不会牺牲显著的灵敏度。为了获得最佳的检测性能，强烈建议 DM 试验的总数（计算方式为 `(dm_high - dm_low) / dm_step`）和总 DM 范围（`dm_high - dm_low`）都超过 512（对于FRB）。

---

## 步骤 4: 执行流水线

数据和配置文件准备就绪后，运行 `astroflow`：

```bash
astroflow /path/to/fast_prex_search.yaml
```

流水线将开始处理 `fast_prex_candidate.csv` 中指定的文件。您将在终端中看到进度日志。

---

## 步骤 5: 分析输出

完成后，`path/to/fast_prex_output/` 目录将包含搜索的所有结果。目录结构如下：

```
.
├── astroflow.log
├── background/
├── cached/
├── candidate/
├── detect/
└── frb/
```

以下是每个目录内容的详细说明：

-   **`astroflow.log`**: 详细记录整个执行过程的综合日志文件。这是查找错误或警告的首选位置。
-   **`background/`**: 包含被归类为射频干扰（RFI）的候选体。这些信号的色散量（DM）和到达时间（TOA）都**不匹配**基准真相中的候选体。
-   **`cached/`**: 存储中间数据产品，以加快重新运行的速度。
-   **`candidate/`**: 包含DM和TOA都**匹配**输入候选体列表中基准真相的候选体。这些代表成功恢复的已知脉冲。
-   **`detect/`**: 包含新检测到的脉冲，其DM**匹配**已知源，但TOA**不匹配**。这些可能是来自同一源的先前未知的脉冲。
-   **`frb/`**: 包含一个由高置信度检测构建的精选数据集。此数据的结构适合用于对AI检测模型进行**微调**。
