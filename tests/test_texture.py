"""
Unit tests for src/modules_2photon.py

All tests use synthetic numpy arrays — no real images required.
Run with: pytest tests/test_texture.py -v
"""
import numpy as np
import pytest

from modules_2photon import (
	binarize_image,
	make_filtered_image,
	convolve,
	make_image_gradients,
	make_structure_tensor_2d,
	make_coherence,
	make_orientation,
	percentage_area,
	perform_statistical_analysis,
	load_pandas_dataframe,
)


# ── binarize_image ────────────────────────────────────────────────────────────

def test_binarize_image_shape(synthetic_shg):
	result = binarize_image(synthetic_shg)
	assert result.shape == synthetic_shg.shape


def test_binarize_image_binary(synthetic_shg):
	result = binarize_image(synthetic_shg)
	unique = set(np.unique(result.astype(int)))
	assert unique.issubset({0, 1})


def test_binarize_image_rejects_3d():
	with pytest.raises(ValueError):
		binarize_image(np.zeros((10, 10, 3), dtype=np.uint8))


# ── make_filtered_image ───────────────────────────────────────────────────────

def test_make_filtered_image_shape(synthetic_shg):
	result = make_filtered_image(synthetic_shg, filter_sigma=2)
	assert result.shape == synthetic_shg.shape


def test_make_filtered_image_smooths(synthetic_shg):
	# Gaussian filter should reduce the standard deviation (smooth variation)
	result = make_filtered_image(synthetic_shg, filter_sigma=3)
	assert result.std() <= synthetic_shg.astype(float).std() + 1.0


# ── convolve ─────────────────────────────────────────────────────────────────

def test_convolve_shape(synthetic_shg):
	kernel = np.ones((5, 5), dtype=np.float32) / 25
	result = convolve(synthetic_shg.astype(np.float32), kernel)
	assert result.shape == synthetic_shg.shape


def test_convolve_uniform_image():
	# Convolving a constant image with any kernel must return the same constant
	img = np.full((64, 64), 128.0, dtype=np.float32)
	kernel = np.ones((7, 7), dtype=np.float32) / 49
	result = convolve(img, kernel)
	np.testing.assert_allclose(result, 128.0, atol=1e-3)


# ── make_image_gradients ──────────────────────────────────────────────────────

def test_make_image_gradients_shapes(synthetic_shg):
	filtered = make_filtered_image(synthetic_shg, 2)
	gx, gy = make_image_gradients(filtered)
	assert gx.shape == synthetic_shg.shape
	assert gy.shape == synthetic_shg.shape


def test_make_image_gradients_uniform():
	# Gradient of a uniform image must be zero everywhere
	img = np.full((64, 64), 100.0, dtype=np.float32)
	gx, gy = make_image_gradients(img)
	np.testing.assert_allclose(gx, 0.0, atol=1e-5)
	np.testing.assert_allclose(gy, 0.0, atol=1e-5)


# ── make_structure_tensor_2d ──────────────────────────────────────────────────

def test_make_structure_tensor_2d_output_shapes(synthetic_shg):
	filtered = make_filtered_image(synthetic_shg, 2)
	gx, gy = make_image_gradients(filtered)
	ST, EV, EVec, Jxx, Jxy, Jyy = make_structure_tensor_2d(gx, gy, local_sigma=10)
	for arr in (Jxx, Jxy, Jyy):
		assert arr.shape == synthetic_shg.shape


# ── make_coherence ────────────────────────────────────────────────────────────

def test_make_coherence_range(synthetic_shg):
	filtered = make_filtered_image(synthetic_shg, 2)
	gx, gy = make_image_gradients(filtered)
	ST, EV, EVec, Jxx, Jxy, Jyy = make_structure_tensor_2d(gx, gy, 10)
	coherence = make_coherence(filtered, EV, ST, threshold_value=5)
	valid = coherence[~np.isnan(coherence)]
	assert valid.min() >= 0.0 - 1e-6
	assert valid.max() <= 1.0 + 1e-6


def test_make_coherence_shape(synthetic_shg):
	filtered = make_filtered_image(synthetic_shg, 2)
	gx, gy = make_image_gradients(filtered)
	ST, EV, EVec, Jxx, Jxy, Jyy = make_structure_tensor_2d(gx, gy, 10)
	coherence = make_coherence(filtered, EV, ST, threshold_value=5)
	assert coherence.shape == synthetic_shg.shape


# ── make_orientation ──────────────────────────────────────────────────────────

def test_make_orientation_range(synthetic_shg):
	filtered = make_filtered_image(synthetic_shg, 2)
	gx, gy = make_image_gradients(filtered)
	ST, EV, EVec, Jxx, Jxy, Jyy = make_structure_tensor_2d(gx, gy, 10)
	orientation = make_orientation(filtered, Jxx, Jxy, Jyy, threshold_value=5)
	valid = orientation[~np.isnan(orientation)]
	assert valid.min() >= 0.0 - 1e-4
	assert valid.max() <= 180.0 + 1e-4


def test_make_orientation_shape(synthetic_shg):
	filtered = make_filtered_image(synthetic_shg, 2)
	gx, gy = make_image_gradients(filtered)
	ST, EV, EVec, Jxx, Jxy, Jyy = make_structure_tensor_2d(gx, gy, 10)
	orientation = make_orientation(filtered, Jxx, Jxy, Jyy, threshold_value=5)
	assert orientation.shape == synthetic_shg.shape


# ── perform_statistical_analysis ─────────────────────────────────────────────

def test_perform_statistical_analysis_shape():
	coherence = np.array([0.2] * 60 + [0.8] * 40, dtype=np.float32)
	result = perform_statistical_analysis("test.tif", 10, None, coherence, 55.0)
	assert result.shape == (1, 4)


def test_perform_statistical_analysis_percentages_sum():
	coherence = np.array([0.1] * 30 + [0.9] * 70, dtype=np.float32)
	result = perform_statistical_analysis("test.tif", 10, None, coherence, 40.0)
	low_pct  = float(result[0, 2])
	high_pct = float(result[0, 3])
	assert abs(low_pct + high_pct - 100.0) < 1.0


def test_perform_statistical_analysis_filename_stored():
	coherence = np.array([0.5] * 100, dtype=np.float32)
	result = perform_statistical_analysis("my_image.tif", 10, None, coherence, 30.0)
	assert result[0, 0] == "my_image.tif"


def test_perform_statistical_analysis_ignores_nan():
	coherence = np.array([0.2, 0.8, np.nan, 0.3, 0.7], dtype=np.float32)
	# Should not raise
	result = perform_statistical_analysis("test.tif", 10, None, coherence, 20.0)
	assert result.shape == (1, 4)


# ── load_pandas_dataframe ─────────────────────────────────────────────────────

def test_load_pandas_dataframe_columns():
	results = np.array([["test.tif", "25.0", "35.0", "65.0"]])
	df = load_pandas_dataframe(results)
	expected = ["Fibrotic percentage [%]", "% Low Coherance regions", "% High Coherance regions"]
	assert list(df.columns) == expected


def test_load_pandas_dataframe_row_count():
	results = np.array([
		["a.tif", "10.0", "40.0", "60.0"],
		["b.tif", "20.0", "55.0", "45.0"],
	])
	df = load_pandas_dataframe(results)
	assert len(df) == 2


# ── percentage_area ───────────────────────────────────────────────────────────

def test_percentage_area_all_nonzero():
	img = np.ones((10, 10), dtype=np.float32)
	assert abs(percentage_area(img) - 100.0) < 1e-4


def test_percentage_area_half():
	img = np.ones((10, 10), dtype=np.float32)
	img[:5, :] = np.nan
	result = percentage_area(img)
	assert abs(result - 50.0) < 1.0
