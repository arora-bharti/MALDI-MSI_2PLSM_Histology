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
- **Batch Processing**: Analyze multiple images with progress tracking
- **Publication-ready Visualizations**: 6-panel mosaic plots

## Repository Structure

```
MALDI-MSI_2PLSM_Histology/
├── src/                          # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── modules_2photon.py        # 2PLSM texture analysis functions
│   └── modules_histo.py          # Histology/nuclei analysis functions
│
├── notebooks/                    # Jupyter notebooks
│   ├── Collagen_textureanalysis.ipynb
│   ├── Nuclei_segmentation.ipynb
│   └── Low_high_coherence_percentage.ipynb
│
├── scripts/                      # Utility scripts
│   └── Merging.ijm               # ImageJ macro for overlay
│
├── app/                          # Streamlit web app (planned)
│   └── .streamlit/config.toml
│
├── sample_data/                  # Sample images for testing
├── tests/                        # Unit tests (planned)
│
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

### 1. Collagen Texture Analysis (2PLSM)

```python
from src.modules_2photon import (
    make_filtered_image,
    make_image_gradients,
    make_structure_tensor_2d,
    make_coherence,
    make_orientation
)
import numpy as np
from PIL import Image

# Load image
raw_image = np.array(Image.open("your_image.tif").convert("L"))

# Parameters
filter_sigma = 2           # Gaussian filter sigma
local_sigma = 10           # Structure tensor local sigma
threshold = np.median(raw_image)

# Analysis pipeline
filtered_image = make_filtered_image(raw_image, filter_sigma)
grad_x, grad_y = make_image_gradients(filtered_image)
structure_tensor, eigenvalues, eigenvectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(
    grad_x, grad_y, local_sigma
)
coherence = make_coherence(filtered_image, eigenvalues, structure_tensor, threshold)
orientation = make_orientation(filtered_image, Jxx, Jxy, Jyy, threshold)
```

Or run the notebook:
```bash
jupyter notebook notebooks/Collagen_textureanalysis.ipynb
```

### 2. Nuclei Segmentation (Histology)

```python
from src.modules_histo import (
    MyNormalizer,
    mean_filter,
    normalize_density_maps,
    weighted_kde_density_map
)
from stardist.models import StarDist2D
from tifffile import imread
import numpy as np

# Load image
img = imread("your_histology.tif")

# Initialize StarDist model
model = StarDist2D.from_pretrained('2D_versatile_he')

# Normalize
mi, ma = np.percentile(img, [25.0, 85.0])
normalizer = MyNormalizer(mi, ma)

# Segment nuclei
labels, polys = model.predict_instances(img, normalizer=normalizer)

# Calculate density
density = mean_filter(labels)
density_normalized = normalize_density_maps(density)
```

Or run the notebook:
```bash
jupyter notebook notebooks/Nuclei_segmentation.ipynb
```

### 3. ImageJ Overlay (Optional)

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

## Related Repositories

- [PyTextureAnalysis](https://github.com/ajinkya-kulkarni/PyTextureAnalysis) - Streamlit app for texture analysis
- [PyHistology](https://github.com/ajinkya-kulkarni/PyHistology) - Streamlit app for histology segmentation
- [PySpatialHistologyAnalysis](https://github.com/ajinkya-kulkarni/PySpatialHistologyAnalysis) - Spatial analysis of H&E images

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
