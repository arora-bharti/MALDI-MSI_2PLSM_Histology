# MALDI-MSI_2PLSM_Histology

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs44303--024--00041--3-blue)](https://doi.org/10.1038/s44303-024-00041-3)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Multimodal image analysis pipeline combining **MALDI mass spectrometry imaging (MALDI-MSI)**, **two-photon laser scanning microscopy (2PLSM)**, and **histology** to characterize the spatial heterogeneity of colorectal cancer (CRC) patient tissues.

![Imaging Protocol](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs44303-024-00041-3/MediaObjects/44303_2024_41_Fig1_HTML.png)

*Figure 1: Imaging protocol for correlation of 2PLSM, MALDI MSI and histology.*

## Overview

This repository contains the source code for:
- **Collagen texture analysis** from 2PLSM images (coherence, orientation, local density)
- **Nuclei segmentation and density analysis** from H&E histology images using StarDist
- **MALDI-MSI analysis** — open-source Python pipeline replacing SCiLS Lab (TIC normalisation, peak alignment, ROC analysis, PCA, bisecting k-means segmentation)

The approach correlates:
- Collagen fiber coherence (organized vs. chaotic regions) from 2PLSM
- Nuclei distribution (high vs. low density regions) from H&E histology
- Proteomics signatures from MALDI-MSI

## Features

- **Structure Tensor Analysis**: Calculate local fiber orientation and coherence
- **Coherence Mapping**: Quantify tissue organization (0=chaotic, 1=organized)
- **Orientation Mapping**: Determine local fiber direction (0-180 degrees)
- **Nuclei Segmentation**: Deep learning-based detection using StarDist
- **Density Heatmaps**: Kernel density estimation of nuclei distribution
- **Nuclear Morphometry**: Eccentricity and area quantification
- **Spatial Nuclei Analysis**: Proximity network graphs and Voronoi tessellation
- **H&E Stain Normalisation**: Ruifrok & Johnston colour deconvolution for batch consistency
- **MALDI-MSI Pipeline**: imzML import, TIC normalisation, peak detection, ROC analysis, PCA, k-means segmentation
- **Batch Processing**: Analyze multiple images with progress tracking
- **Streamlit Web App**: Interactive browser-based interface for all modalities
- **Publication-ready Visualizations**: 6-panel mosaic plots

## Repository Structure

```
MALDI-MSI_2PLSM_Histology/
├── src/                          # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── modules_2photon.py        # 2PLSM texture analysis functions
│   ├── modules_histo.py          # Histology/nuclei analysis functions
│   └── modules_maldi.py          # MALDI-MSI analysis functions
│
├── notebooks/                    # Jupyter notebooks
│   ├── Collagen_textureanalysis.ipynb
│   ├── Nuclei_segmentation.ipynb
│   ├── Low_high_coherence_percentage.ipynb
│   ├── Orientation_vectorfield.ipynb
│   └── MALDI_analysis.ipynb
│
├── scripts/                      # CLI scripts & utilities
│   ├── analyze_texture.py        # CLI for texture analysis
│   ├── segment_nuclei.py         # CLI for nuclei segmentation
│   └── Merging.ijm               # ImageJ macro for overlay
│
├── app/                          # Streamlit web app
│   └── streamlit_app.py
│
├── .streamlit/                   # Streamlit configuration
│   └── config.toml
│
├── tests/                        # Unit tests (pytest)
│   ├── conftest.py               # Shared synthetic fixtures
│   ├── test_texture.py           # Tests for modules_2photon
│   ├── test_histo.py             # Tests for modules_histo
│   └── test_maldi.py             # Tests for modules_maldi
│
├── sample_data/                  # Sample images
├── requirements.txt              # Python dependencies
├── CITATION.cff                  # Citation metadata
├── LICENSE                       # AGPL-3.0 license
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/arora-bharti/MALDI-MSI_2PLSM_Histology.git
cd MALDI-MSI_2PLSM_Histology
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

The easiest way to use this pipeline is via command-line scripts:

#### Collagen Texture Analysis
```bash
# Analyze a single image
python scripts/analyze_texture.py -i image.tif -o results/

# Analyze a folder of images
python scripts/analyze_texture.py -i data/2plsm_images/ -o results/

# With custom parameters
python scripts/analyze_texture.py -i data/ -o results/ \
    --filter-sigma 3 \
    --local-sigma 15 \
    --local-density-kernel 25
```

#### Nuclei Segmentation
```bash
# Segment a single image
python scripts/segment_nuclei.py -i histology.tif -o results/

# Segment a folder of images
python scripts/segment_nuclei.py -i data/histology/ -o results/

# With custom thresholds
python scripts/segment_nuclei.py -i data/ -o results/ \
    --prob-thresh 0.5 \
    --nms-thresh 0.4
```

#### MALDI-MSI Analysis
```bash
# Interactive step-by-step MALDI analysis
jupyter notebook notebooks/MALDI_analysis.ipynb
```

### Interactive Analysis (Notebooks)

For interactive exploration and visualization:

```bash
# Texture analysis with step-by-step visualization
jupyter notebook notebooks/Collagen_textureanalysis.ipynb

# Nuclei segmentation with morphometry
jupyter notebook notebooks/Nuclei_segmentation.ipynb

# Coherence statistics
jupyter notebook notebooks/Low_high_coherence_percentage.ipynb

# Orientation vector field overlay
jupyter notebook notebooks/Orientation_vectorfield.ipynb
```

### Streamlit Web App

A browser-based interface covering all three modalities:

```bash
streamlit run app/streamlit_app.py
```

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests for modules requiring TensorFlow/StarDist are automatically skipped if those libraries are not installed.

### Python API

For integration into your own scripts:

```python
# Texture analysis
from src.modules_2photon import (
    make_filtered_image, make_structure_tensor_2d,
    make_coherence, make_orientation
)

# Nuclei segmentation
from src.modules_histo import (
    MyNormalizer, mean_filter, normalize_density_maps
)
```

See the notebooks for complete usage examples.

### ImageJ Overlay (Optional)

For merging coherence/density maps with original images:

1. Open ImageJ/Fiji
2. Run `scripts/Merging.ijm`
3. Adjust input/output paths as needed

## Parameters Guide

### Texture Analysis Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `filter_sigma` | Gaussian smoothing sigma | 1-5 |
| `local_sigma` | Structure tensor local window | 5-20 |
| `threshold` | Intensity threshold for masking | Auto (median) |
| `local_density_kernel` | Kernel size for density calculation | 10-30 |

### Nuclei Segmentation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `prob_thresh` | Detection probability threshold | 0.4 |
| `nms_thresh` | Non-maximum suppression threshold | 0.3 |
| `normalization_percentiles` | Input normalization range | [25, 85] |
| `num_bins` | Quantile bins for morphometry | 6 |

## Output Files

### Texture Analysis
- `FilteredImage_*.tif` - Gaussian filtered image
- `DensityImage_*.tif` - Local density map
- `CoheranceImage_*.tif` - Coherence heatmap (0-1)
- `OrientationImage_*.tif` - Orientation map (0-180 degrees)
- `Results_*.png` - 6-panel visualization

### Nuclei Segmentation
- `labeled/*.tif` - Segmented nuclei labels
- `density/*.tif` - Nuclear density map
- `circularity/*.tif` - Eccentricity-binned map
- `area/*.tif` - Area-binned map
- `packaging/*.tif` - Nuclear packing density (KDE)
- `*_mosaic.png` - 6-panel visualization
- `results/*.csv` - Quantitative metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@article{arora2024maldi,
  title={MALDI imaging combined with two-photon microscopy reveals local differences in the heterogeneity of colorectal cancer},
  author={Arora, Bharti and Kulkarni, Ajinkya and Markus, M. Andrea and Ramos-Gomes, Fernanda and Bohnenberger, Hanibal and Str{\"o}bel, Philipp and Alves, Frauke and Klein, Oliver},
  journal={npj Imaging},
  volume={2},
  number={35},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s44303-024-00041-3}
}
```

## License

This project is licensed under the **GNU Affero General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

## Authors

- **Bharti Arora** - [bharti.arora@mpinat.mpg.de](mailto:bharti.arora@mpinat.mpg.de)
- **Ajinkya Kulkarni** - [ajinkya.kulkarni@mpinat.mpg.de](mailto:ajinkya.kulkarni@mpinat.mpg.de)

Max Planck Institute for Multidisciplinary Sciences, Göttingen, Germany

## Acknowledgements

This research was funded by:
- European Union's Horizon 2020 (Marie Sklodowska-Curie grant No 857894 - CAST)
- Ministry for Science and Culture of Lower Saxony ("Agile, bio-inspired architectures" - ABA)
- Federal Ministry of Education and Research (MSTAR #031L0220A)
- BCRT and Berlin Institute of Health
