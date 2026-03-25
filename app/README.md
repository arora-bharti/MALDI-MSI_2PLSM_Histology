# Streamlit Web Application

This folder is reserved for the interactive Streamlit web application.

## Planned Features

- **Tab 1**: Collagen Texture Analysis (2PLSM)
  - Upload 2D grayscale image
  - Adjust parameters (filter sigma, local window, structure tensor sigma)
  - View coherence heatmap, orientation map, density map
  - Download results

- **Tab 2**: Nuclei Segmentation (Histology)
  - Upload H&E stained image
  - Adjust StarDist detection thresholds
  - View labeled nuclei, density heatmap
  - Download results

- **Tab 3**: Batch Processing
  - Process multiple images at once
  - Download zipped results

## Running the App

```bash
# Install Streamlit
pip install streamlit

# Run the app
streamlit run app/streamlit_app.py
```
