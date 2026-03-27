"""
Unit tests for src/modules_histo.py

Tests only functions with no TensorFlow/StarDist dependency.
The entire file is skipped gracefully if heavy ML libraries are not installed.
Run with: pytest tests/test_histo.py -v
"""
import numpy as np
import pytest

# Skip the whole file if modules_histo or its heavy dependencies cannot be imported
try:
	from modules_histo import (
		normalize_density_maps,
		normalize_array,
		normalize_staining,
		make_nuclei_network,
		make_voronoi_tessellation,
		mean_filter,
	)
except ImportError as e:
	pytest.skip(f"modules_histo dependencies not available: {e}", allow_module_level=True)


# ── normalize_density_maps ────────────────────────────────────────────────────

def test_normalize_density_maps_max_is_one():
	arr = np.random.rand(64, 64).astype(np.float32) * 5
	result = normalize_density_maps(arr)
	assert abs(float(np.nanmax(result)) - 1.0) < 1e-6


def test_normalize_density_maps_shape():
	arr = np.random.rand(32, 32).astype(np.float32)
	assert normalize_density_maps(arr).shape == (32, 32)


def test_normalize_density_maps_zero_input():
	arr = np.zeros((10, 10), dtype=np.float32)
	result = normalize_density_maps(arr)
	# All-zero input: divide by zero is protected, should return NaN or zero
	assert not np.any(np.isinf(result))


# ── normalize_array ───────────────────────────────────────────────────────────

def test_normalize_array_range():
	arr = np.random.rand(100, 100).astype(np.float32) * 1000
	result = normalize_array(arr)
	assert float(result.min()) >= 0.0
	assert float(result.max()) <= 255.0 + 1e-4


def test_normalize_array_shape():
	arr = np.random.rand(50, 80).astype(np.float32)
	assert normalize_array(arr).shape == (50, 80)


# ── normalize_staining ────────────────────────────────────────────────────────

def test_normalize_staining_shape():
	# Construct a plausible H&E-like image (pink background + some purple areas)
	img = np.full((64, 64, 3), [240, 180, 210], dtype=np.uint8)
	img[20:30, 20:30] = [100, 60, 150]
	result = normalize_staining(img)
	assert result.shape == img.shape


def test_normalize_staining_dtype():
	img = np.full((64, 64, 3), [200, 150, 180], dtype=np.uint8)
	result = normalize_staining(img)
	assert result.dtype == np.uint8


def test_normalize_staining_values_clipped():
	img = np.full((64, 64, 3), 128, dtype=np.uint8)
	result = normalize_staining(img)
	assert int(result.min()) >= 0
	assert int(result.max()) <= 255


# ── mean_filter ───────────────────────────────────────────────────────────────

def test_mean_filter_shape(synthetic_label_image):
	result = mean_filter(synthetic_label_image)
	assert result.shape == synthetic_label_image.shape


def test_mean_filter_nonnegative(synthetic_label_image):
	result = mean_filter(synthetic_label_image)
	assert float(result.min()) >= 0.0


# ── make_nuclei_network ───────────────────────────────────────────────────────

def test_make_nuclei_network_node_count(synthetic_label_image):
	G, centroids = make_nuclei_network(synthetic_label_image)
	# Fixture has exactly 10 labelled circles
	assert G.number_of_nodes() == 10


def test_make_nuclei_network_centroids_shape(synthetic_label_image):
	G, centroids = make_nuclei_network(synthetic_label_image)
	assert centroids.ndim == 2
	assert centroids.shape[1] == 2


def test_make_nuclei_network_edge_weights_positive(synthetic_label_image):
	G, _ = make_nuclei_network(synthetic_label_image)
	for u, v, data in G.edges(data=True):
		assert data["weight"] > 0.0


def test_make_nuclei_network_max_distance(synthetic_label_image):
	# With max_distance=1, no edges should form (nuclei are far apart)
	G, _ = make_nuclei_network(synthetic_label_image, max_distance=1)
	assert G.number_of_edges() == 0


# ── make_voronoi_tessellation ─────────────────────────────────────────────────

def test_make_voronoi_tessellation_point_count(synthetic_label_image):
	vor, centroids = make_voronoi_tessellation(synthetic_label_image)
	assert len(vor.points) == 10


def test_make_voronoi_tessellation_centroids_shape(synthetic_label_image):
	vor, centroids = make_voronoi_tessellation(synthetic_label_image)
	assert centroids.shape == (10, 2)


def test_make_voronoi_tessellation_too_few_nuclei():
	# Should raise ValueError with fewer than 4 nuclei
	tiny = np.zeros((64, 64), dtype=np.int32)
	tiny[10:15, 10:15] = 1
	tiny[30:35, 30:35] = 2
	with pytest.raises(ValueError, match="at least 4"):
		make_voronoi_tessellation(tiny)
