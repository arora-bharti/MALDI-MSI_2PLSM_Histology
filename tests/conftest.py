"""
Shared pytest fixtures — all synthetic, no real patient data required.
"""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_shg():
	"""
	256×256 uint8 grayscale image with horizontal sinusoidal stripes.
	Simulates an SHG collagen image with organised fibres (high coherence).
	Fixed seed ensures deterministic results across test runs.
	"""
	rng = np.random.default_rng(42)
	h, w = 256, 256
	y = np.arange(h)[:, None]
	x = np.arange(w)[None, :]

	# Horizontal stripes — high coherence, orientation ≈ 90°
	stripes = 128 + 80 * np.sin(2 * np.pi * y / 20)
	# Low-amplitude noise
	noise = rng.normal(0, 8, (h, w))

	img = np.clip(stripes + noise, 0, 255).astype(np.uint8)
	return img


@pytest.fixture(scope="session")
def synthetic_label_image():
	"""
	256×256 int32 label image with exactly 10 non-overlapping circular nuclei
	at known centroid positions.
	"""
	img = np.zeros((256, 256), dtype=np.int32)
	centres = [
		(30,  30), (30,  90), (30, 150), (30, 210),
		(90,  30), (90,  90), (90, 150), (90, 210),
		(150, 30), (150, 90),
	]
	radius = 12
	for label_id, (cy, cx) in enumerate(centres, start=1):
		yy, xx = np.ogrid[:256, :256]
		mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
		img[mask] = label_id
	return img


@pytest.fixture(scope="session")
def synthetic_spectra():
	"""
	50 spectra × 200 m/z bins, float32.
	Spectra 0–24 have a strong peak at bin 100 (group A).
	Spectra 25–49 have no peak at bin 100 (group B).
	ROC AUC for bin 100 should be ≈ 1.0.
	"""
	rng = np.random.default_rng(0)
	mat = rng.random((50, 200)).astype(np.float32) * 0.01
	# Group A: clear signal at bin 100
	mat[:25, 100] = rng.random(25).astype(np.float32) * 0.5 + 0.5
	return mat
