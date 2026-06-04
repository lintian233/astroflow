# AstroFlow Manual

AstroFlow is a GPU-accelerated single-pulse and fast radio burst search pipeline for FILTERBANK (`.fil`) and PSRFITS (`.fits`) observations. It combines CUDA dedispersion, RFI mitigation, AI-assisted candidate detection, and candidate plotting behind a YAML-driven command-line workflow.

```{image} cand.gif
:alt: AstroFlow candidate preview
:width: 100%
```

This is the official AstroFlow manual. The online documentation is published at [lintian233.github.io/astroflow](https://lintian233.github.io/astroflow/), and the latest source version can be found at [github.com/lintian233/astroflow](https://github.com/lintian233/astroflow).

## Contents

```{toctree}
:maxdepth: 2

installation
quick_start
configuration
fast_prex
dataset_model_release
```

## Overview

- CUDA-accelerated dedispersion and search over configurable DM, time, and frequency grids.
- AI candidate detection with the bundled `yolov11n` model or a user-provided model path.
- RFI mitigation through external masks, zero-DM filtering, and GPU-accelerated IQRM.
- Batch, directory, monitor, and dataset-validation modes.
- Candidate plots, logs, and optional CSV candidate summaries.
- Public training dataset and separately versioned model checkpoints.


## Citation

If AstroFlow is useful in your research, cite:

**ASTROFLOW: A Real-Time End-to-End Pipeline for Radio Single-Pulse Searches**  
[https://doi.org/10.3847/1538-4365/ae4a26](https://doi.org/10.3847/1538-4365/ae4a26)

## Project Links

- Documentation: [https://lintian233.github.io/astroflow/](https://lintian233.github.io/astroflow/)
- Source: [https://github.com/lintian233/astroflow](https://github.com/lintian233/astroflow)
- Issues: [https://github.com/lintian233/astroflow/issues](https://github.com/lintian233/astroflow/issues)
