# Streamlit Web Application

Interactive browser-based interface for all three imaging modalities.

## Features

- **Tab 1 — Collagen Texture Analysis (2PLSM)**: Upload a grayscale SHG image, adjust filter sigma, local window, and structure tensor parameters, view coherence heatmap and orientation map, download results CSV.
- **Tab 2 — Nuclei Segmentation (Histology)**: Upload an H&E image, adjust StarDist detection thresholds, view labelled nuclei and density heatmap, download results.
- **Tab 3 — Batch Processing**: Upload multiple images, run texture analysis across all, download a combined results table.

## Running the App

```bash
streamlit run app/streamlit_app.py
```
