# Processing the FAST PREX FRB Dataset with AstroFlow

This tutorial provides a comprehensive, step-by-step guide to processing the FAST PREX dataset using the `astroflow` pipeline. We will cover data acquisition, pipeline configuration, execution, and interpretation of the results.

This dataset comprises observations containing known FRBs with precursor emissions, making it an excellent validation set for single-pulse and FRB search pipelines.

## Prerequisites

- **AstroFlow Installation**: Ensure `astroflow` is installed and executable. See the [installation guide](./README.md#methods).
- **Disk Space**: Approximately **200 GB** of free disk space is required for the dataset.
- **Dependencies**: `wget` and standard Unix command-line tools.

---

## Step 1: Download the Dataset

The dataset is hosted on the Science Data Bank.

### 1.1. Obtain Data URLs

First, we need a list of URLs for all the files in the dataset.

1.  Navigate to the dataset's homepage: [FAST PREX Dataset](https://www.scidb.cn/en/detail?dataSetId=3b3cf2f75a74419b89a56cc9626af2a0)
2.  At the bottom of the page, click the **`Get All URLs`** button.
3.  Save the complete list of URLs into a local text file named `fast_prex.txt`.

### 1.2. Execute Batch Download

We will use `xargs` and `wget` to perform a parallel download, which significantly speeds up the process. The following command downloads up to 8 files concurrently.

```bash
# Create a directory for the data
mkdir -p fast_prex_data && cd fast_prex_data

# extract FRBs url and save as fast_prex_frbs.txt
cat fast_prex.txt | grep FRB > fast_prex_frbs.txt

# Run the parallel download
xargs -P 8 -n 1 -I {} bash -c '
  url="$1"
  # Extract filename from URL, handling different URL formats
  filename=$(printf "%s" "$url" | sed -n "s/.*[?&]fileName=\([^&]*\).*/\1/p")
  filename=${filename:-$(basename "${url%%\?*}")}
  
  # Download with resume (-c) and a clean progress bar
  echo "Downloading: $filename"
  wget -c --show-progress --progress=bar:force:noscroll -O "$filename" "$url"
' _ {} < fast_prex_frbs.txt
```

This script iterates through each URL in `fast_prex.txt`:
- **`-P 8`**: Runs up to 8 `wget` processes in parallel.
- **`-c`**: Allows `wget` to resume interrupted downloads.
- **`-O "$filename"`**: Saves the file with the correct name extracted from the URL.

After the download completes, verify the contents of the `fast_prex_data/` directory.

---

## Step 2: Prepare Auxiliary Files

A predefined candidate list is required for `dataset` mode. This file contains the ground truth information (file, DM, TOA) for known FRBs in the dataset.

```bash
wget -O fast_prex_candidate.csv https://raw.githubusercontent.com/lintian233/astroflow/main/docs/candidates/fast_prex_candidate.csv
```

**Note:** This file is **not** used to guide the detection process. Instead, it serves as a ground truth reference after the search is complete. The pipeline uses this list to cross-match the candidates it finds against the known FRBs, allowing for the automatic calculation of performance metrics such as recall and precision. The detection parameters are configured independently in the YAML file.

---

## Step 3: Configure the Search Pipeline

Create a YAML configuration file named `fast_prex_search.yaml`. This file instructs `astroflow` on how to process the data.

```yaml
# save as fast_prex_search.yaml

# --- I/O Configuration ---
input: path/to/fast_prex_data/      # Directory containing the downloaded FITS files
output: path/to/fast_prex_output/   # Directory for all pipeline outputs
mode: dataset                       # `dataset` mode processes files based on a candidate list
candpath: /path/to/fast_prex_candidate.csv # Path to the candidate CSV file

# --- Processing & Performance ---
dedgpu: 0                 # GPU device ID for dedispersion
detgpu: 0                 # GPU device ID for AI detection (use a different ID for multi-GPU)
cputhread: 32             # Number of CPU threads for I/O and pre-processing
plotworker: 16            # Number of parallel processes for generating candidate plots
# onlycand: True          # Optional: Disable candidate plotting for benchmarking (reduces I/O) (TestPypi current)

# --- Detection Parameters ---
modelname: yolov11n       # AI model for detection.
# modelpath: yolo11n_0816_v1.pt # Optional: path to a custom-trained model
confidence: 0.4           # Detection confidence threshold (0.0 to 1.0)
timedownfactor: 8         # Time series down-sampling factor before detection. Increases sensitivity to wider pulses.

# --- RFI Mitigation ---
rfi: 
  use_mask: 0             # (0/1) Set to 1 to use external static channel masks
  use_iqrm: 1             # (0/1) Enable the GPU-accelerated IQRM algorithm
  use_zero_dm: 0          # (0/1) Enable zero-DM scrubbing (often disabled for FRB searches)

iqrm:
  mode: 0                 # 0=mean, 1=std. Statistical method for outlier detection.
  radius_frac: 0.1        # Radius fraction for lag selection in the autocorrelation function.
  nsigma: 7.0             # Threshold for flagging RFI. A higher value is more conservative.
  geofactor: 1.5          # Geometric factor for lag progression.
  win_sec: 0              # Window size in seconds (0 for full data).
  hop_sec: 6.04           # Hop size in seconds for sliding window analysis.
  include_tail: true      # Process the remaining data chunk at the end.

# maskdir: /path/to/FAST_PREFIX_MASK # Optional: Directory of .bad_chans files if use_mask=1

# --- Search Space ---
tsample:
  - name: t0
    t: 0.5 # seconds. The duration of each data chunk processed.

dmrange:
  - name: dm100_700
    dm_low: 100
    dm_high: 700
    dm_step: 1 # pc cm^-3. A step of 1 is typical for FRB searches at high DMs.

freqrange:
  - name: Lband
    freq_start: 1000
    freq_end: 1499.5 # MHz. Full L-band coverage for FAST.

# --- Plotting Configuration ---
dmtconfig:
  minpercentile: 0
  maxpercentile: 100
  meadianbulr: 1 3
  guassion: 1 5

specconfig:
  minpercentile: 0    
  maxpercentile: 100
  tband: 120 # ms. Time window for the dynamic spectrum plot.
  mode: subband # 'subband' or 'standard' or 'detrend'
  dtrend: True # optional: per-subband detrending (default false)
  norm: False   # optional: per-subband normalization (default true)
  subfreq: 256 # optional: number of subbands (default 128)
  subtsamp: 4  # optional: time binning factor (default 4)
  savetype: png # optional: image format png/jpg
```

Optional spectrum controls give you finer control over the plotting step—comment out any of the extra lines above to fall back to the defaults baked into the plotter.

**Key Parameter Explanations:**
- **`mode: dataset`**: This mode is designed to process a list of data files. It performs a blind search on each file and then uses the ground truth information (DM, time) from the `candpath` file to automatically classify the detected candidates (e.g., as `candidate`, `detect`, or `background`) and calculate performance metrics. It is ideal for validation, performance testing, and characterization of the pipeline.
- **`timedownfactor: 8`**: FAST data has a high time resolution. Down-sampling by a factor of 8 makes the pipeline more sensitive to temporally broader pulses, which is common for scattered FRBs, while also reducing computational load.
- **`dm_step: 1`**: At high dispersion measures (DM > 100 pc cm⁻³), the dispersion smearing within a single channel becomes larger than the smearing between adjacent DM trials with a step of 1.0. This makes it a computationally efficient choice without sacrificing significant sensitivity. For optimal detection performance, it is strongly recommended that the total number of DM trials (calculated as `(dm_high - dm_low) / dm_step`) and the total DM range (`dm_high - dm_low`) both exceed 512(for FRBs).

> **Performance Tip**: The FAST_PREX task is I/O bound. If you want to benchmark this task, it is recommended to enable `onlycand: True` in the YAML configuration. This setting disables the candidate plotting module, avoiding unnecessary I/O and making the benchmark results more stable. (TestPypi Current)

---

## Step 4: Execute the Pipeline

With the data and configuration file ready, run `astroflow`:

```bash
astroflow /path/to/fast_prex_search.yaml
```

The pipeline will begin processing the files specified in `fast_prex_candidate.csv`. You will see progress logs in your terminal.

---

## Step 5: Analyze the Output

Upon completion, the `path/to/fast_prex_output/` directory will be populated with all the results of the search. The directory structure will look something like this:

```
.
├── astroflow.log
├── background/
├── cached/
├── candidate/
├── detect/
└── frb/
```

Here is a detailed explanation of each directory's contents:

-   **`astroflow.log`**: A comprehensive log file detailing the entire execution process. It's the first place to look for errors or warnings.
-   **`background/`**: Contains candidates classified as Radio Frequency Interference (RFI). These are signals where both the Dispersion Measure (DM) and Time of Arrival (TOA) do **not** match the ground truth candidates.
-   **`cached/`**: Stores intermediate info products.
-   **`candidate/`**: Contains candidates where both the DM and TOA **match** the ground truth from the input candidate list. These represent the successfully recovered known pulses.
-   **`detect/`**: Contains newly detected pulses where the DM **matches** a known source, but the TOA does **not**. These could be previously unknown pulses from the same source.
-   **`frb/`**: Contains a curated dataset constructed from the high-confidence detections. This data is structured to be suitable for **fine-tuning** the AI detection model with new, real-world examples.






