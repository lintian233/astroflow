# YAML Configuration Reference

AstroFlow is configured with one YAML file passed to the `astroflow` command:

```bash
astroflow search.yaml
```

Most searches need four groups of settings: input/output, compute resources, search grid, and plotting.

## Minimal Single-File Example

```yaml
input: FRB180417.fil
output: frb180417
mode: single

dedgpu: 0
detgpu: 0
cputhread: 8
plotworker: 1

modelname: yolov11n
confidence: 0.4
timedownfactor: 1
gencand: true

rfi:
  use_mask: false
  use_zero_dm: false
  use_iqrm: true

iqrm:
  mode: 1
  radius_frac: 0.1
  nsigma: 10.0
  geofactor: 1.5
  win_sec: 0
  hop_sec: 0.5
  include_tail: true

tsample:
  - name: t0
    t: 0.5

dm_limt:
  - name: dm_all
    dm_low: 0
    dm_high: 800

dmrange:
  - name: dm_10_600
    dm_low: 10
    dm_high: 600
    dm_step: 1

freqrange:
  - name: fullband
    freq_start: 1130
    freq_end: 1465

dmtconfig:
  minpercentile: 0
  maxpercentile: 99.99

specconfig:
  minpercentile: 0.1
  maxpercentile: 99.9
  tband: 120
  mode: subband
  subfreq: 128
  subtsamp: 2
  dtrend: false
  norm: true
  savetype: png
```

## Top-Level Options

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `input` | string | Yes | none | Input `.fil`/`.fits` file, input directory, or dataset directory depending on `mode`. |
| `output` | string | Yes | none | Output directory for logs, cached products, candidate images, and candidate CSV files. Created automatically. |
| `mode` | string | Yes | none | Processing mode. One of `single`, `directory`, `muti`, `monitor`, `dataset`. `muti` is retained as a compatibility alias for directory processing. |
| `candpath` | string | Dataset mode only | none | Candidate CSV path for ground-truth matching in `dataset` mode. |
| `psrfitsbackend` | string | No | `cpp` | PSRFITS reader backend. One of `cpp`, `python`. |
| `gencand` | boolean | No | `false` | Write `astroflow_cands.csv` candidate summaries beside candidate outputs. |
| `onlycand` | boolean | No | `false` | Save candidate metrics without generating full candidate plots. Useful when metrics are needed but image products are not. |
| `fastcand` | boolean | No | `false` | Use the lightweight candidate-output path and reduce plotting overhead. Recommended for high-throughput benchmarks. |
| `include_last` | boolean | No | `false` | Include the final partial processing chunk when enabled. |
| `minfileage` | number | No | `0` | In monitor mode, only process files older than this many seconds. Helps avoid reading files that are still being written. |

## Processing And GPU Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `dedgpu` | integer | `0` | GPU device index for dedispersion. |
| `detgpu` | integer | `0` | GPU device index for AI detection. |
| `cputhread` | integer | `16` | CPU worker/thread count. Also sets `OMP_NUM_THREADS`. |
| `plotworker` | integer | `2` | Number of plotting worker processes. |
| `batchsize` | integer | `128` | Detection batch size. |
| `timedownfactor` | integer or null | none | Time-axis downsampling factor passed into search configuration. Set explicitly for normal runs. |

## Detection Options

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `modelname` | string | Yes | none | Detector model name. Currently only `yolov11n` is supported for standard searches. |
| `modelpath` | string | No | auto-downloaded bundled model | Path to model weights. If omitted, AstroFlow downloads and verifies the default YOLO model under `~/.config/astroflow`. |
| `confidence` | number | Recommended | none | Detector confidence threshold. Typical value: `0.4`. |
| `snrhold` | number | No | `-100` | SNR threshold for candidate filtering. |

## Search Grid Options

### `tsample`

List of time chunk definitions.

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Label used in output names/logs. |
| `t` | number | Time span in seconds for each detection slice. |

```yaml
tsample:
  - name: t0
    t: 0.5
```

Shorter `t` values can help narrow pulses but increase the number of chunks; wider values are often more robust for broader or scattered pulses.

### `dmrange`

Actual DM trial grid used for dedispersion.

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Label used in output names/logs. |
| `dm_low` | number | Lower DM bound in pc cm^-3. |
| `dm_high` | number | Upper DM bound in pc cm^-3. |
| `dm_step` | number | DM trial spacing. |

```yaml
dmrange:
  - name: dm_10_600
    dm_low: 10
    dm_high: 600
    dm_step: 1
```

For FRB searches, use enough trials to cover the expected source DM and keep the number of DM trials large enough for stable detector behavior.

### `dm_limt`

Candidate DM acceptance ranges. The configuration key is `dm_limt` for compatibility with existing AstroFlow configuration files.

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Label for the candidate DM range. |
| `dm_low` | number | Lower candidate DM bound. |
| `dm_high` | number | Upper candidate DM bound. |

```yaml
dm_limt:
  - name: valid_frb_dm
    dm_low: 0
    dm_high: 800
```

If omitted, the detector receives `None` for the candidate DM limits.

### `freqrange`

Frequency windows in MHz.

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Label used in output names/logs. |
| `freq_start` | number | Start frequency in MHz. |
| `freq_end` | number | End frequency in MHz. |

```yaml
freqrange:
  - name: fullband
    freq_start: 1130
    freq_end: 1465
```

Multiple `tsample`, `dmrange`, and `freqrange` entries are combined as a Cartesian product, so the number of runs grows quickly.

## RFI Options

```yaml
rfi:
  use_mask: false
  use_zero_dm: false
  use_iqrm: true

iqrm:
  mode: 1
  radius_frac: 0.1
  nsigma: 10.0
  geofactor: 1.5
  win_sec: 0
  hop_sec: 0.5
  include_tail: true
```

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `rfi.use_mask` | boolean | `false` | Apply external bad-channel masks. Requires `maskfile` or `maskdir`. |
| `rfi.use_zero_dm` | boolean | `false` | Apply zero-DM filtering. This can suppress broadband RFI but may affect bright astrophysical signals. |
| `rfi.use_iqrm` | boolean | `false` unless set | Enable IQRM-based RFI mitigation. |
| `maskfile` | string | empty | Single bad-channel mask file. The file must exist when configured. |
| `maskdir` | string | none | Directory containing masks named like `<basename>_your_rfi_mask.bad_chans`. |

### `iqrm`

`iqrm` is required when `rfi.use_iqrm: true` and must contain every field below.

| Field | Type | Description |
| --- | --- | --- |
| `mode` | integer | IQRM statistic mode. Use `0` for mean-based statistics and `1` for the alternate robust statistic supported by the backend. |
| `radius_frac` | number | Radius as a fraction of channel count for lag selection. |
| `nsigma` | number | Outlier threshold. Higher values are more conservative. |
| `geofactor` | number | Geometric progression factor for lag selection. |
| `win_sec` | number | Window length in seconds. `0` means use the full data block. |
| `hop_sec` | number | Sliding-window hop in seconds. |
| `include_tail` | boolean | Process the remaining tail window when the data length is not an exact multiple of the hop. |

## Plotting Options

### `dmtconfig`

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `minpercentile` | number | `0.1` | Lower percentile for DM-time image scaling. |
| `maxpercentile` | number | `99.99` | Upper percentile for DM-time image scaling. |

Older examples may include `meadianbulr` and `guassion`. These options are preserved for backward compatibility; new configurations only need the percentile controls above.

### `specconfig`

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `minpercentile` | number | `0.1` | Lower percentile for dynamic-spectrum image scaling. |
| `maxpercentile` | number | `99.99` | Upper percentile for dynamic-spectrum image scaling. |
| `tband` | number | `300.0` | Dynamic-spectrum time window in milliseconds. |
| `mode` | string | `subband` | Spectrum display mode. Supported values are `subband`, `standard`, and `detrend`; `std` is normalized to `standard`. |
| `subtsamp` | integer | `4` | Time aggregation factor used in subband plotting. |
| `subfreq` | integer | `128` | Number of output frequency subbands. Values less than or equal to zero use all channels. |
| `dtrend` | boolean | `false` | Apply linear detrending in `subband` mode. Alias: `detrend`. |
| `norm` | boolean | `true` | Normalize subbands before plotting. |
| `savetype` | string | `png` | Candidate image format. Use `png` or `jpg`. |
| `snr_boxcar_max_ms` | number or null | `30.0` | Maximum boxcar width for SNR estimation in milliseconds. Alias: `boxcar_max_ms`. Use `null` for the default 30-sample path, or `<= 0` to disable the cap. |
| `onlyspec` | boolean | `false` | Save spectrum-only output instead of the full candidate panel in supported plotting paths. |
| `gc_collect_every_files` | integer | `10` | Run garbage collection after this many plotted files. Use `0` to disable. |

## Mode Behavior

| Mode | `input` must be | Behavior |
| --- | --- | --- |
| `single` | File | Search one `.fil` or `.fits` file. |
| `directory` | Directory | Search supported files in a directory. |
| `muti` | Directory | Compatibility alias for multi-file directory search. |
| `monitor` | Directory | Watch a directory and process new files. CLI options `--check-interval` and `--stop-file` control polling. |
| `dataset` | Directory | Search files and compare detections with `candpath` ground truth. |

## Output Layout

A typical run creates:

```text
output/
├── astroflow.log
├── background/
├── cached/
├── candidate/
├── detect/
├── frb/
└── astroflow_cands.csv
```

The exact folders depend on the selected mode and whether candidate plotting or CSV generation is enabled.
