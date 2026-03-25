# Tests

This folder is reserved for unit tests.

## Planned Tests

- `test_texture_analysis.py` - Tests for coherence, orientation calculations
- `test_nuclei_analysis.py` - Tests for nuclei segmentation functions
- `test_io.py` - Tests for file I/O operations

## Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=src
```
