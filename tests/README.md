# Tests

Unit tests for all three source modules. All tests use synthetic data — no real patient images required.

## Test files

- `test_texture.py` — 18 tests for `modules_2photon` (coherence, orientation, structure tensor, convolution, statistics)
- `test_histo.py` — 14 tests for `modules_histo` (stain normalisation, nuclei network, Voronoi); skipped gracefully if TensorFlow/StarDist are not installed
- `test_maldi.py` — 20 tests for `modules_maldi` (TIC normalisation, peak alignment, ion images, PCA, bisecting k-means)

## Running Tests

```bash
pytest tests/ -v

# With coverage
pip install pytest-cov
pytest tests/ --cov=src
```
