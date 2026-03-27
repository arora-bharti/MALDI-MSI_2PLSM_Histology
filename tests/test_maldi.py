"""
Unit tests for src/modules_maldi.py

All tests use synthetic spectral data — no real imzML files required.
Run with: pytest tests/test_maldi.py -v
"""
import numpy as np
import pandas as pd
import pytest

from modules_maldi import (
	normalize_tic,
	find_and_align_peaks,
	generate_ion_image,
	build_data_cube,
	get_discriminative_mz,
	run_pca,
	run_bisecting_kmeans,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mz():
	return np.linspace(800, 3200, 200).astype(np.float32)


def _make_coords(n_spectra=50, width=10):
	return [(i % width + 1, i // width + 1) for i in range(n_spectra)]


def _make_image_shape(n_spectra=50, width=10):
	height = (n_spectra + width - 1) // width
	return (height, width)


# ── normalize_tic ─────────────────────────────────────────────────────────────

def test_normalize_tic_rows_sum_to_one(synthetic_spectra):
	result = normalize_tic(synthetic_spectra)
	row_sums = result.sum(axis=1)
	np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), rtol=1e-5)


def test_normalize_tic_shape_preserved(synthetic_spectra):
	result = normalize_tic(synthetic_spectra)
	assert result.shape == synthetic_spectra.shape


def test_normalize_tic_zero_spectrum_no_nan():
	mat = np.zeros((5, 100), dtype=np.float32)
	mat[0, 50] = 1.0
	result = normalize_tic(mat)
	assert not np.any(np.isnan(result))
	assert not np.any(np.isinf(result))


def test_normalize_tic_nonnegative(synthetic_spectra):
	result = normalize_tic(synthetic_spectra)
	assert float(result.min()) >= 0.0


# ── find_and_align_peaks ──────────────────────────────────────────────────────

def test_find_and_align_peaks_output_shapes(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz, min_intensity_fraction=0.01)
	assert peak_matrix.shape[0] == normed.shape[0]
	assert peak_matrix.shape[1] == len(peak_mz)
	assert len(peak_mz) > 0


def test_find_and_align_peaks_detects_known_peak(synthetic_spectra):
	# synthetic_spectra has a strong signal at bin 100 (m/z ≈ 1560)
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz, min_intensity_fraction=0.001)
	target_mz = mz[100]
	# At least one detected peak should be within 500 ppm of the known signal
	diffs = np.abs(peak_mz - target_mz)
	assert diffs.min() <= target_mz * 500 / 1e6


def test_find_and_align_peaks_nonnegative(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	_, peak_matrix = find_and_align_peaks(normed, mz)
	assert float(peak_matrix.min()) >= 0.0


# ── generate_ion_image ────────────────────────────────────────────────────────

def test_generate_ion_image_shape(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz)
	coords = _make_coords()
	shape = _make_image_shape()
	img = generate_ion_image(peak_matrix, peak_mz, peak_mz[0], coords, shape)
	assert img.shape == shape


def test_generate_ion_image_missing_mz_returns_nan(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz)
	coords = _make_coords()
	shape = _make_image_shape()
	# m/z value far outside the range — all pixels should be NaN
	img = generate_ion_image(peak_matrix, peak_mz, 99999.0, coords, shape)
	assert np.all(np.isnan(img))


def test_generate_ion_image_pixel_value(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz)
	coords = _make_coords()
	shape = _make_image_shape()
	img = generate_ion_image(peak_matrix, peak_mz, peak_mz[0], coords, shape)
	# First spectrum is at (x=1, y=1) → row=0, col=0
	expected = float(peak_matrix[0, 0])
	assert abs(float(img[0, 0]) - expected) < 1e-5


# ── build_data_cube ───────────────────────────────────────────────────────────

def test_build_data_cube_shape(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	_, peak_matrix = find_and_align_peaks(normed, mz)
	coords = _make_coords()
	shape = _make_image_shape()
	cube, mask = build_data_cube(peak_matrix, coords, shape)
	assert cube.shape == (shape[0], shape[1], peak_matrix.shape[1])
	assert mask.shape == shape


def test_build_data_cube_pixel_mask(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	_, peak_matrix = find_and_align_peaks(normed, mz)
	coords = _make_coords()
	shape = _make_image_shape()
	_, mask = build_data_cube(peak_matrix, coords, shape)
	assert mask.sum() == len(coords)


# ── get_discriminative_mz ─────────────────────────────────────────────────────

def test_get_discriminative_mz_filters_correctly():
	df = pd.DataFrame({
		"mz":     [1000.0, 1500.0, 2000.0, 2500.0],
		"auc":    [0.70,   0.50,   0.35,   0.55],
		"pvalue": [0.001,  0.001,  0.001,  0.001],
	})
	result = get_discriminative_mz(df, auc_high=0.6, auc_low=0.4)
	# 0.70 ≥ 0.6 → keep; 0.50 between → discard; 0.35 ≤ 0.4 → keep; 0.55 between → discard
	assert len(result) == 2
	assert set(result["mz"].tolist()) == {1000.0, 2000.0}


def test_get_discriminative_mz_empty_input():
	df = pd.DataFrame({"mz": [], "auc": [], "pvalue": []})
	result = get_discriminative_mz(df)
	assert len(result) == 0


def test_get_discriminative_mz_all_discriminative():
	df = pd.DataFrame({
		"mz":     [1000.0, 2000.0],
		"auc":    [0.9,    0.1],
		"pvalue": [0.001,  0.001],
	})
	result = get_discriminative_mz(df)
	assert len(result) == 2


# ── run_bisecting_kmeans ──────────────────────────────────────────────────────

def test_run_bisecting_kmeans_segment_image_shape(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz)
	disc_df = pd.DataFrame({"mz": [float(peak_mz[0])], "auc": [0.8], "pvalue": [0.001]})
	coords = _make_coords()
	shape = _make_image_shape()
	seg_img, labels, areas = run_bisecting_kmeans(
		peak_matrix, peak_mz, disc_df, coords, shape, n_clusters=2)
	assert seg_img.shape == shape


def test_run_bisecting_kmeans_areas_sum_to_100(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz)
	disc_df = pd.DataFrame({"mz": [float(peak_mz[0])], "auc": [0.8], "pvalue": [0.001]})
	coords = _make_coords()
	shape = _make_image_shape()
	_, _, areas = run_bisecting_kmeans(
		peak_matrix, peak_mz, disc_df, coords, shape, n_clusters=3)
	assert abs(areas["percentage_area"].sum() - 100.0) < 0.1


def test_run_bisecting_kmeans_n_clusters(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz)
	disc_df = pd.DataFrame({"mz": peak_mz[:3].tolist(), "auc": [0.8] * 3, "pvalue": [0.001] * 3})
	coords = _make_coords()
	shape = _make_image_shape()
	_, _, areas = run_bisecting_kmeans(
		peak_matrix, peak_mz, disc_df, coords, shape, n_clusters=4)
	assert len(areas) == 4


# ── run_pca ───────────────────────────────────────────────────────────────────

def test_run_pca_n_components(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz)
	disc_df = pd.DataFrame({"mz": peak_mz[:6].tolist(), "auc": [0.8] * 6, "pvalue": [0.001] * 6})
	coords = _make_coords()
	shape = _make_image_shape()
	pc_images, loadings_df, expl_var, _ = run_pca(
		peak_matrix, peak_mz, disc_df, coords, shape, n_components=3)
	assert len(pc_images) == 3


def test_run_pca_image_shapes(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz)
	disc_df = pd.DataFrame({"mz": peak_mz[:5].tolist(), "auc": [0.8] * 5, "pvalue": [0.001] * 5})
	coords = _make_coords()
	shape = _make_image_shape()
	pc_images, _, _, _ = run_pca(peak_matrix, peak_mz, disc_df, coords, shape, n_components=2)
	for img in pc_images:
		assert img.shape == shape


def test_run_pca_explained_variance_sums_to_one(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz)
	disc_df = pd.DataFrame({"mz": peak_mz[:5].tolist(), "auc": [0.8] * 5, "pvalue": [0.001] * 5})
	coords = _make_coords()
	shape = _make_image_shape()
	_, _, expl_var, _ = run_pca(peak_matrix, peak_mz, disc_df, coords, shape, n_components=5)
	# Explained variance of all components must sum to ≤ 1.0
	assert float(expl_var.sum()) <= 1.0 + 1e-5


def test_run_pca_loadings_columns(synthetic_spectra):
	mz = _make_mz()
	normed = normalize_tic(synthetic_spectra)
	peak_mz, peak_matrix = find_and_align_peaks(normed, mz)
	disc_df = pd.DataFrame({"mz": peak_mz[:4].tolist(), "auc": [0.8] * 4, "pvalue": [0.001] * 4})
	coords = _make_coords()
	shape = _make_image_shape()
	_, loadings_df, _, _ = run_pca(peak_matrix, peak_mz, disc_df, coords, shape, n_components=2)
	assert "mz" in loadings_df.columns
	assert "PC1" in loadings_df.columns
	assert "PC2" in loadings_df.columns
