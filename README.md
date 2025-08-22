<div align="center">

English | [简体中文](./docs/README_zh-CN.md)

<!-- <img src="🌌" width="320px" alt="AstroFlow" /> -->

<h2 id="title">AstroFlow 🌌</h2>

<p>
  <!-- Package & registry badges -->
  <a href="https://pypi.org/project/pulseflow/">
    <img src="https://img.shields.io/pypi/v/pulseflow" alt="PyPI"></a>
  <a href="https://pepy.tech/project/pulseflow">
    <img src="https://static.pepy.tech/badge/pulseflow" alt="Downloads"></a>
  <a href="https://hub.docker.com/r/lintian233/astroflow-build">
    <img src="https://img.shields.io/docker/pulls/lintian233/astroflow-build" alt="Docker pulls"></a>
  <a href="https://github.com/lintian233/astroflow/issues">
    <img src="https://img.shields.io/badge/contributions-welcome-green" alt="Contrib welcome"></a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/lintian233/astroflow" alt="License"></a>
</p>

<!-- <a href="https://trendshift.io/repositories/astroflow" target="_blank">
  <img src="https://trendshift.io/api/badge/repositories/astroflow" alt="AstroFlow | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a> -->

</div>

One stop realtime GPU-accelerated single-pulse/FRB search pipeline

- ⚡ **GPU-Accelerated Processing** - CUDA dedispersion with advanced optimizations (FMA, shared memory, loop unrolling)
- 🤖 **AI-Powered Detection** - YOLOv11n trained on 100k+ sim FRBs dataset with high accuracy
- 🧹 **Complete RFI Mitigation** - RFI removal with PSRFITS & FILTERBANK support  
- ⚙️ **User-Friendly Pipeline** - Simple YAML configuration with Docker deployment and bilingual interface

Feedback: use [Issues](https://github.com/lintian233/astroflow/issues) or [Discussions](https://github.com/lintian233/astroflow/discussions).  
<!-- See [Contribution Guide](./docs/CONTRIBUTING.md) for how to get involved. -->

<h2 id="updates">Updates</h2>


- **Aug 22, 2025** - feat: Add GPU-accelerated RFI marking and decouple CPU/GPU implementations
- **Aug 21, 2025** — Public preview of `astroflow` CLI ; CUDA dedispersion and YOLO-based detector integrated.  
- **Aug 20, 2025** — Docker image (CUDA 12.x, PyTorch) and end-to-end benchmark scripts.  

<!-- > [!NOTE]
> Roadmap and milestones are tracked in [Projects](https://github.com/lintian233/astroflow/projects). -->

<h2 id="preview">Preview</h2>

<div align="center">
  <img src="./docs/cand.gif" width="100%" />
</div>


### Methods

<details open>
  <summary>pip install</summary>

1. Python 3.10–3.12 recommended.

2. Install:
   ```bash   
   pip install pulseflow
   ```

3. Quick start:
   ```bash
   astroflow demo.yaml
   ```

  
</details>

<details>
  <summary>source code</summary>
1. cudatoolkit>=12.0 glibc>=2.27 gcc/gxx>=11.0 ubutnu18.04+.

2. git clone 
    ```bash
    git clone https://github.com/lintian233/astroflow 
    cd astroflow
    ```
3. build from soure
   ```bash
   source configure.sh
   ```
4. pip install
   ```bash
   pip install -e .
   ```

</details>


<h2 id="citation">Citation</h2>

If you use **AstroFlow** in your research, please cite our work:

> [!NOTE]
> Our scientific paper describing the AstroFlow pipeline is currently **in preparation**. 
> We will update this section with the proper paper citation once our manuscript is published. 
> Please check back for updates or watch this repository for announcements.
> In the meantime, please cite this repository:
> 
```bibtex
@misc{astroflow2025,
  title={AstroFlow: GPU-accelerated single-pulse/FRB search pipeline},
  author={lintian233},
  year={2025},
  url={https://github.com/lintian233/astroflow},
  note={Software pipeline for fast radio burst detection and analysis}
}
```

<h2 id="usage">YAML Configuration</h2>

Below are the commonly used YAML configuration options:

<details open>
  <summary>Basic Configuration</summary>

| Option | Description | Example | Required |
|--------|-------------|---------|----------|
| `input` | Input file path or directory | `B0534+2200.fil/fits` | ✓ |
| `output` | Output directory path | `B0534+2200_results` | ✓ |
| `mode` | Processing mode | `single`/`multi`/`monitor`/`dataset` | ✓ |
| `candpath` | Candidate file path (for dataset mode) | `candidates.csv` | Only for dataset mode |

</details>

<details>
  <summary>GPU and Processing</summary>

| Option | Description | Example | Default |
|--------|-------------|---------|---------|
| `dedgpu` | GPU ID for dedispersion | `0` | `0` |
| `detgpu` | GPU ID for detection | `1` | `0` |
| `cputhread` | Number of CPU threads | `64` | `8` |
| `plotworker` | Number of plot workers | `16` | `4` |

</details>

<details>
  <summary>Detection Parameters</summary>

| Option | Description | Example | Default |
|--------|-------------|---------|---------|
| `confidence` | Detection confidence threshold | `0.4` | `0.372` |
| `snrhold` | SNR threshold | `5` | `5` |
| `modelname` | Detector model type | current only `yolov11n`
| `modelpath` | Path to customodel weights | `model.pt` | - |
| `timedownfactor` | Time downsampling factor | `8` | `1` |

</details>

<details>
  <summary>Time and DM Configuration</summary>

| Option | Description | Example |
|--------|-------------|---------|
| `tsample` | Time sampling configurations | See below |
| `dm_limt` | DM limit ranges | See below |
| `dmrange` | DM search ranges | See below |

```yaml
tsample:
  - name: t0
    t: 0.5  # seconds (for single pulse)
```

```yaml
dm_limt:
  - name: limt1
    dm_low: 50
    dm_high: 60
  - name: limt4
    dm_low: 100
    dm_high: 700
```

```yaml
dmrange:
  - name: dm1
    dm_low: 50
    dm_high: 60
    dm_step: 0.018
  - name: dm3
    dm_low: 80
    dm_high: 700
    dm_step: 1
```

</details>

<details>
  <summary>Frequency Configuration</summary>

| Option | Description | Example |
|--------|-------------|---------|
| `freqrange` | Frequency ranges for processing | See below |

```yaml
freqrange:
  - name: freq1
    freq_start: 1030  # MHz
    freq_end: 1445    # MHz (full band)
  - name: freq2
    freq_start: 1030
    freq_end: 1250    # MHz (sub-band 1)
  - name: freq3
    freq_start: 1250
    freq_end: 1445    # MHz (sub-band 2)
    # For FAST data
  - name: freq3
    freq_start: 1000
    freq_end: 1499.5  # MHz
```

</details>

<details>
  <summary>RFI Mitigation</summary>

| Option | Description | Example |
|--------|-------------|---------|
| `rfi` | RFI mitigation method | `ai`/`your`/`mixup` |
| `maskfile` | Single RFI mask file | `/path/to/file` |
| `maskdir` | Directory containing RFI masks | `/path/to/RFI_MASK/` |

Example RFI configuration:
```yaml
rfi: ai
maskfile: /path/to/maskdfile.bad_chans
# OR use directory for multiple files
maskdir: /path/to/maskdir
```

</details>

<!-- <details>
  <summary>Data Processing</summary>

| Option | Description | Example |
|--------|-------------|---------|
| `preprocess` | Preprocessing pipeline | See below |

```yaml
preprocess:
  - clip: 0.01                # Clipping threshold
  - meadianbulr: 1 3         # Median blur parameters
  - guassion: 1 5            # Gaussian filter parameters
```

</details> -->

<details>
  <summary>Plotting Configuration</summary>

| Option | Description | Example |
|--------|-------------|---------|
| `dmtconfig` | DMT Plot Config | See below |
```yaml
dmtconfig:
  minpercentile: 0           # Minimum percentile for scaling
  maxpercentile: 100         # Maximum percentile (99.9 for FAST data)
  meadianbulr: 1 3          # Median blur for DMT plot
  guassion: 1 5             # Gaussian filter for DMT plot
```


| Option | Description | Example |
|--------|-------------|---------|
| `specconfig` | Spectrum Plot Config | See below |

```yaml
specconfig:
  minpercentile: 0          # Minimum percentile for scaling
  maxpercentile: 100        # Maximum percentile for scaling
  tband: 20                 # Time band in ms (120 for FAST data)
  mode: subband             # Plot mode: subband/standard
```

</details>

<details>
  <summary>Real Examples</summary>

**Single Pulsar Observation (Crab):**
```yaml
input: B0534+2200_20250413_194909_ant01p0.fil 
output: B0534+2200_20250413_194909
mode: single

timedownfactor: 1
confidence: 0.4
dedgpu: 0
detgpu: 1
cputhread: 32

snrhold: 5
modelname: yolov11n
# use default
# modelpath: yolo11n_0816_v1.pt 

rfi: ai
maskfile: file.bad_chans

tsample:
  - name: t0
    t: 0.05

dm_limt:
  - name: limt1
    dm_low: 50
    dm_high: 60

dmrange:
  - name: dm1
    dm_low: 50
    dm_high: 60
    dm_step: 0.018

dmtconfig:
  minpercentile: 0
  maxpercentile: 99.9
  meadianbulr: 1 3
  guassion: 1 5

specconfig:
  minpercentile: 0    
  maxpercentile: 100
  tband: 10 #ms
  mode: subband # subband/standard

```

**FAST_PREFIX Dataset Processing:**
```yaml
input: FAST_PREFIX_DATA
output: FAST_PREFIX
mode: dataset

candpath: fast_prefix_candidate.csv
timedownfactor: 8
confidence: 0.372

dedgpu: 0
detgpu: 1
cputhread: 64

modelname: yolov11n
# use default
# modelpath: yolo11n_0816_v1.pt 
plotworker: 16

rfi: ai
maskdir: FAST_PREFIX_RFI_MASK

tsample:
  - name: t2
    t: 0.5

dm_limt:
  - name: limt4
    dm_low: 100
    dm_high: 700

dmrange:
  - name: dm3
    dm_low: 80
    dm_high: 700
    dm_step: 1

freqrange:
  - name: freq3
    freq_start: 1000
    freq_end: 1499.5


dmtconfig:
  minpercentile: 0
  maxpercentile: 99.9
  meadianbulr: 1 3
  guassion: 1 5

specconfig:
  minpercentile: 0    
  maxpercentile: 100
  tband: 120 #ms
  mode: standard # subband/standard

```
</details>

<h2 id="contrib">Contributors</h2>

<a href="https://github.com/lintian233/astroflow/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=lintian233/astroflow"/" />
</a>

![Alt](https://repobeats.axiom.co/api/embed/b68167f9b82d6200ed0da3f95fa021d1d989d978.svg "Repobeats analytics image")

<h2 id="star_hist">Star History</h2>

<a href="https://www.star-history.com/#lintian233/astroflow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=lintian233/astroflow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=lintian233/astroflow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=lintian233/astroflow&type=Date" />
 </picture>
</a>