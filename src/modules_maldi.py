#!/usr/bin/env python3
# encoding: utf-8
#
# Copyright (C) 2024 Max Planck Institute for Multidisclplinary Sciences
# Copyright (C) 2024 Bharti Arora <bharti.arora@mpinat.mpg.de>
# Copyright (C) 2024 Ajinkya Kulkarni <ajinkya.kulkarni@mpinat.mpg.de>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

########################################################################################

# This file contains all modules/functions for MALDI-MSI data analysis,
# replicating the SCiLS Lab 2024a Pro workflow from the paper using open-source tools.
#
# Paper: "MALDI imaging combined with two-photon microscopy reveals local differences
# in the heterogeneity of colorectal cancer", npj Imaging 2, 35 (2024).
# DOI: https://doi.org/10.1038/s44303-024-00041-3
#
# SCiLS Lab steps replicated here:
#   1. TIC normalisation (maximal interval processing mode, no smoothing)
#   2. Peak finding and alignment across all spectra
#   3. QuPath region annotation import (open-source alternative to .sef / QuPath-SCiLS plugin)
#   4. ROC analysis per m/z between annotated tissue regions (AUC >= 0.6 or <= 0.4, p < 0.01)
#   5. PCA on discriminative m/z values
#   6. Bisecting k-means segmentation on discriminative m/z values
#
# Prerequisites:
#   pip install pyimzml shapely

########################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import json
import os

from scipy import signal
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

########################################################################################

def read_imzml(filepath):
	"""
	Read an imzML file and return the intensity matrix, m/z axis, pixel coordinates,
	and the reconstructed image shape.

	The imzML format is the open community standard for MALDI-MSI data exchange.
	Export from Bruker instruments via FlexImaging 5.1: File → Export → imzML.
	The .imzML (XML header) and .ibd (binary data) files must reside in the same folder.

	Requires: pip install pyimzml

	Parameters:
	filepath (str): Path to the .imzML file.

	Returns:
	tuple:
		mz_array (np.ndarray): 1D array of m/z values (common axis).
		intensity_matrix (np.ndarray): Float32 array of shape (n_spectra, n_mz).
		coordinates (list of tuple): (x, y) pixel coordinates, one per spectrum.
		image_shape (tuple): (height, width) of the tissue section in pixels.
	"""
	try:
		from pyimzml.ImzMLParser import ImzMLParser
	except ImportError:
		raise ImportError("pyimzml is required. Install with: pip install pyimzml")

	parser = ImzMLParser(filepath)
	coordinates = [(int(x), int(y)) for x, y, *_ in parser.coordinates]

	xs = [c[0] for c in coordinates]
	ys = [c[1] for c in coordinates]
	image_shape = (max(ys), max(xs))

	# Read all spectra; use the first spectrum's m/z as the common axis
	all_spectra = []
	for idx in tqdm(range(len(coordinates)), desc='Reading spectra', leave=False):
		mzs, intensities = parser.getspectrum(idx)
		all_spectra.append((mzs, intensities))

	mz_array = all_spectra[0][0].astype(np.float32)
	n_mz = len(mz_array)
	n_spectra = len(coordinates)

	intensity_matrix = np.zeros((n_spectra, n_mz), dtype=np.float32)
	for idx, (mzs, intensities) in enumerate(all_spectra):
		if len(intensities) == n_mz:
			intensity_matrix[idx] = intensities.astype(np.float32)
		else:
			# Profile mode with varying length: interpolate onto common axis
			intensity_matrix[idx] = np.interp(mz_array, mzs, intensities,
											   left=0.0, right=0.0).astype(np.float32)

	return mz_array, intensity_matrix, coordinates, image_shape

########################################################################################

def normalize_tic(intensity_matrix):
	"""
	Apply Total Ion Count (TIC) normalisation to a spectral intensity matrix.

	Each spectrum is divided by its summed intensity so that all spectra integrate
	to 1.0. This replicates the TIC normalisation setting used in SCiLS Lab
	(maximal interval processing mode, TIC normalisation, no smoothing).

	Parameters:
	intensity_matrix (np.ndarray): Raw intensity matrix of shape (n_spectra, n_mz).

	Returns:
	np.ndarray: TIC-normalised intensity matrix of shape (n_spectra, n_mz), float32.
	"""
	tic = intensity_matrix.sum(axis=1, keepdims=True).astype(np.float32)
	tic = np.where(tic == 0, 1.0, tic)
	return intensity_matrix / tic

########################################################################################

def find_and_align_peaks(intensity_matrix, mz_array, min_intensity_fraction=0.01, ppm_tolerance=100.0):
	"""
	Detect peaks in the mean spectrum and integrate each spectrum within a ppm window
	around every peak centre.

	This replicates SCiLS Lab 'maximal interval processing mode': peaks are found in
	the pooled mean spectrum (medium noise reduction = 1% intensity threshold) and
	each individual spectrum is summed within ±ppm_tolerance Da of each peak.
	No smoothing is applied, consistent with the paper settings.

	Parameters:
	intensity_matrix (np.ndarray): TIC-normalised matrix of shape (n_spectra, n_mz).
	mz_array (np.ndarray): 1D m/z axis of length n_mz.
	min_intensity_fraction (float): Peaks below this fraction of the maximum mean
		intensity are rejected (medium noise reduction). Default: 0.01.
	ppm_tolerance (float): Integration window in parts-per-million around each peak
		centre. Default: 100 ppm (typical for Bruker rapifleX reflector mode).

	Returns:
	tuple:
		peak_mz (np.ndarray): 1D array of aligned peak m/z values (n_peaks,).
		peak_matrix (np.ndarray): Float32 array of shape (n_spectra, n_peaks)
			containing integrated intensity at each peak for each spectrum.
	"""
	mean_spectrum = intensity_matrix.mean(axis=0)
	min_height = float(mean_spectrum.max()) * min_intensity_fraction

	peak_indices, _ = signal.find_peaks(mean_spectrum, height=min_height, distance=3)
	peak_mz = mz_array[peak_indices]

	n_spectra = intensity_matrix.shape[0]
	n_peaks = len(peak_mz)
	peak_matrix = np.zeros((n_spectra, n_peaks), dtype=np.float32)

	for j, mz_center in enumerate(tqdm(peak_mz, desc='Aligning peaks', leave=False)):
		delta = float(mz_center) * ppm_tolerance / 1e6
		window = (mz_array >= mz_center - delta) & (mz_array <= mz_center + delta)
		if window.any():
			peak_matrix[:, j] = intensity_matrix[:, window].sum(axis=1)

	return peak_mz, peak_matrix

########################################################################################

def generate_ion_image(peak_matrix, peak_mz, target_mz, coordinates, image_shape, ppm_tolerance=100.0):
	"""
	Generate a 2D ion image for a single m/z value.

	Parameters:
	peak_matrix (np.ndarray): Aligned intensity matrix of shape (n_spectra, n_peaks).
	peak_mz (np.ndarray): 1D array of aligned peak m/z values.
	target_mz (float): Target m/z value to visualise.
	coordinates (list of tuple): (x, y) pixel coordinates for each spectrum.
	image_shape (tuple): (height, width) of the tissue image in pixels.
	ppm_tolerance (float): Search window in ppm to locate the nearest peak. Default: 100.

	Returns:
	np.ndarray: 2D float32 ion image of shape (height, width).
		Pixels with no spectrum are set to NaN.
	"""
	ion_image = np.full(image_shape, np.nan, dtype=np.float32)
	delta = float(target_mz) * ppm_tolerance / 1e6
	mz_diffs = np.abs(peak_mz - target_mz)
	candidates = np.where(mz_diffs <= delta)[0]

	if len(candidates) == 0:
		return ion_image

	best_idx = int(candidates[np.argmin(mz_diffs[candidates])])
	for spectrum_idx, (x, y) in enumerate(coordinates):
		ion_image[y - 1, x - 1] = peak_matrix[spectrum_idx, best_idx]

	return ion_image

########################################################################################

def build_data_cube(peak_matrix, coordinates, image_shape):
	"""
	Reconstruct the full 3D data cube (height × width × n_peaks) from sparse coordinates.

	Parameters:
	peak_matrix (np.ndarray): Aligned intensity matrix of shape (n_spectra, n_peaks).
	coordinates (list of tuple): (x, y) pixel coordinates for each spectrum.
	image_shape (tuple): (height, width) of the tissue image.

	Returns:
	tuple:
		data_cube (np.ndarray): 3D float32 array of shape (height, width, n_peaks).
		pixel_mask (np.ndarray): Boolean 2D array of shape (height, width),
			True where a spectrum was acquired.
	"""
	height, width = image_shape
	n_peaks = peak_matrix.shape[1]
	data_cube = np.zeros((height, width, n_peaks), dtype=np.float32)
	pixel_mask = np.zeros((height, width), dtype=bool)

	for idx, (x, y) in enumerate(coordinates):
		data_cube[y - 1, x - 1, :] = peak_matrix[idx]
		pixel_mask[y - 1, x - 1] = True

	return data_cube, pixel_mask

########################################################################################

def import_qupath_annotations(geojson_path, image_shape):
	"""
	Import tissue region annotations exported from QuPath as GeoJSON and rasterise
	them into binary pixel masks.

	In the paper, H&E images were annotated in QuPath by a pathologist, then
	transferred to SCiLS Lab as proprietary .sef files via the QuPath-SCiLS plugin.
	This function provides the open-source equivalent:

	  QuPath → Annotations → Export as GeoJSON (File menu or via script)
	  → load here with image_shape matching the H&E scan pixel dimensions.

	Requires: pip install shapely

	Parameters:
	geojson_path (str): Path to the QuPath GeoJSON annotation export.
	image_shape (tuple): (height, width) of the tissue image in pixels.

	Returns:
	dict: {annotation_name (str): binary_mask (np.ndarray bool, shape H×W)}.
		Each mask is True at pixels belonging to that annotated region class.
	"""
	try:
		from shapely.geometry import shape
		from shapely.vectorized import contains
	except ImportError:
		raise ImportError("shapely is required. Install with: pip install shapely")

	with open(geojson_path, 'r') as f:
		geojson = json.load(f)

	height, width = image_shape
	region_masks = {}

	for feature in geojson.get('features', []):
		props = feature.get('properties', {})
		classification = props.get('classification', {})
		name = classification.get('name', props.get('name', 'Unknown'))
		if name not in region_masks:
			region_masks[name] = np.zeros((height, width), dtype=bool)
		geom = shape(feature['geometry'])
		ys, xs = np.mgrid[0:height, 0:width]
		inside = contains(geom, xs.ravel().astype(float), ys.ravel().astype(float))
		region_masks[name] |= inside.reshape(height, width)

	return region_masks

########################################################################################

def compute_roc_per_mz(peak_matrix, peak_mz, mask_a, mask_b, coordinates, image_shape):
	"""
	Compute ROC AUC for each aligned m/z between two annotated tissue regions.

	Replicates the supervised ROC analysis from the paper: spectra from region A
	are labelled 1, spectra from region B are labelled 0, and sklearn's roc_auc_score
	is computed per m/z. Statistical significance is assessed with the Mann-Whitney U
	test. Discriminative m/z values have AUC >= 0.6 or <= 0.4 and p < 0.01.

	Parameters:
	peak_matrix (np.ndarray): Aligned intensity matrix of shape (n_spectra, n_peaks).
	peak_mz (np.ndarray): 1D array of aligned peak m/z values.
	mask_a (np.ndarray): Boolean 2D mask (H×W) for region A (labelled 1).
	mask_b (np.ndarray): Boolean 2D mask (H×W) for region B (labelled 0).
	coordinates (list of tuple): (x, y) pixel coordinates for each spectrum.
	image_shape (tuple): (height, width) of the tissue image.

	Returns:
	pd.DataFrame: Columns ['mz', 'auc', 'pvalue'] for all tested m/z values,
		sorted by AUC descending. Only rows with p < 0.01 are returned.
	"""
	height, width = image_shape
	labels = np.full(len(coordinates), -1, dtype=np.int8)

	for idx, (x, y) in enumerate(coordinates):
		row, col = y - 1, x - 1
		if 0 <= row < height and 0 <= col < width:
			if mask_a[row, col]:
				labels[idx] = 1
			elif mask_b[row, col]:
				labels[idx] = 0

	valid = labels >= 0
	y_true = labels[valid].astype(int)
	X = peak_matrix[valid]

	if y_true.sum() == 0 or (y_true == 0).sum() == 0:
		raise ValueError("One or both regions contain no spectra within the provided masks.")

	results = []
	for j in tqdm(range(len(peak_mz)), desc='ROC analysis per m/z', leave=False):
		scores = X[:, j]
		try:
			auc = roc_auc_score(y_true, scores)
			_, pvalue = mannwhitneyu(scores[y_true == 1], scores[y_true == 0],
									 alternative='two-sided')
			results.append({'mz': float(peak_mz[j]), 'auc': float(auc), 'pvalue': float(pvalue)})
		except Exception:
			continue

	df = pd.DataFrame(results)
	if df.empty:
		return df
	df = df[df['pvalue'] < 0.01].copy()
	df = df.sort_values('auc', ascending=False).reset_index(drop=True)
	return df

########################################################################################

def get_discriminative_mz(roc_df, auc_high=0.6, auc_low=0.4):
	"""
	Filter ROC results to retain discriminative m/z values.

	Applies the thresholds from the paper: AUC >= auc_high (higher in region A)
	or AUC <= auc_low (higher in region B).

	Parameters:
	roc_df (pd.DataFrame): Output of compute_roc_per_mz().
	auc_high (float): Upper AUC threshold. Default: 0.6.
	auc_low (float): Lower AUC threshold. Default: 0.4.

	Returns:
	pd.DataFrame: Filtered DataFrame of discriminative m/z values.
	"""
	mask = (roc_df['auc'] >= auc_high) | (roc_df['auc'] <= auc_low)
	return roc_df[mask].reset_index(drop=True)

########################################################################################

def run_pca(peak_matrix, peak_mz, discriminative_mz_df, coordinates, image_shape, n_components=5):
	"""
	Run PCA on the subset of discriminative m/z values and generate spatial PC maps.

	Replicates the PCA step from the paper: PCA was performed on discriminative
	peptide m/z values (AUC >= 0.6 or <= 0.4). The first three PCs accounted for
	80.34% of variability. Each PC score is projected back to pixel coordinates to
	produce a spatial intensity map.

	Parameters:
	peak_matrix (np.ndarray): Aligned intensity matrix of shape (n_spectra, n_peaks).
	peak_mz (np.ndarray): 1D array of aligned peak m/z values.
	discriminative_mz_df (pd.DataFrame): Output of get_discriminative_mz().
	coordinates (list of tuple): (x, y) pixel coordinates for each spectrum.
	image_shape (tuple): (height, width) of the tissue image.
	n_components (int): Number of principal components to compute. Default: 5.

	Returns:
	tuple:
		pc_images (list of np.ndarray): Spatial maps, one float32 2D array per PC.
		loadings_df (pd.DataFrame): m/z loadings per PC, columns ['mz', 'PC1', ...].
		explained_variance (np.ndarray): Fraction of variance explained by each PC.
		pca (sklearn.decomposition.PCA): Fitted PCA object for further use.
	"""
	disc_mz_values = discriminative_mz_df['mz'].values
	col_indices = np.array([int(np.argmin(np.abs(peak_mz - mz))) for mz in disc_mz_values])

	X = peak_matrix[:, col_indices]
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	pca = PCA(n_components=n_components)
	scores = pca.fit_transform(X_scaled)

	height, width = image_shape
	pc_images = []
	for pc in range(n_components):
		img = np.full((height, width), np.nan, dtype=np.float32)
		for idx, (x, y) in enumerate(coordinates):
			img[y - 1, x - 1] = float(scores[idx, pc])
		pc_images.append(img)

	loadings_df = pd.DataFrame(
		pca.components_.T,
		index=disc_mz_values,
		columns=[f'PC{i + 1}' for i in range(n_components)]
	)
	loadings_df.index.name = 'mz'
	loadings_df = loadings_df.reset_index()

	return pc_images, loadings_df, pca.explained_variance_ratio_, pca

########################################################################################

def run_bisecting_kmeans(peak_matrix, peak_mz, discriminative_mz_df, coordinates, image_shape, n_clusters=6):
	"""
	Segment the tissue by bisecting k-means clustering on discriminative m/z values.

	Replicates the bisecting k-means segmentation from the paper. The paper used 6
	segments (default here) and quantified the percentage area occupied by each cluster.
	Clustering is performed only on discriminative peptide m/z values identified
	by ROC analysis.

	Requires scikit-learn >= 1.1 for BisectingKMeans.

	Parameters:
	peak_matrix (np.ndarray): Aligned intensity matrix of shape (n_spectra, n_peaks).
	peak_mz (np.ndarray): 1D array of aligned peak m/z values.
	discriminative_mz_df (pd.DataFrame): Output of get_discriminative_mz().
	coordinates (list of tuple): (x, y) pixel coordinates for each spectrum.
	image_shape (tuple): (height, width) of the tissue image.
	n_clusters (int): Number of segments. Default: 6 (as in the paper).

	Returns:
	tuple:
		segment_image (np.ndarray): 2D int32 array of shape (height, width).
			Values 0 to n_clusters-1 indicate cluster assignment.
			Pixels with no spectrum are set to -1.
		cluster_labels (np.ndarray): 1D int array of cluster labels per spectrum.
		cluster_areas (pd.DataFrame): Columns ['cluster', 'n_spectra', 'percentage_area'].
	"""
	disc_mz_values = discriminative_mz_df['mz'].values
	col_indices = np.array([int(np.argmin(np.abs(peak_mz - mz))) for mz in disc_mz_values])
	X = peak_matrix[:, col_indices]

	bkm = BisectingKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
	cluster_labels = bkm.fit_predict(X)

	height, width = image_shape
	segment_image = np.full((height, width), -1, dtype=np.int32)
	for idx, (x, y) in enumerate(coordinates):
		segment_image[y - 1, x - 1] = int(cluster_labels[idx])

	total = len(coordinates)
	cluster_areas = pd.DataFrame([
		{'cluster': k + 1,
		 'n_spectra': int((cluster_labels == k).sum()),
		 'percentage_area': 100.0 * float((cluster_labels == k).sum()) / total}
		for k in range(n_clusters)
	])

	return segment_image, cluster_labels, cluster_areas

########################################################################################

def plot_mean_spectrum(mz_array, intensity_matrix, peak_mz=None, title='Mean Spectrum'):
	"""
	Plot the mean TIC-normalised spectrum with optional peak annotations.

	Parameters:
	mz_array (np.ndarray): 1D m/z axis.
	intensity_matrix (np.ndarray): Normalised intensity matrix (n_spectra, n_mz).
	peak_mz (np.ndarray or None): Detected peak m/z values to mark. Default: None.
	title (str): Figure title.

	Returns:
	matplotlib.figure.Figure
	"""
	mean_spectrum = intensity_matrix.mean(axis=0)

	fig, ax = plt.subplots(figsize=(14, 4))
	ax.plot(mz_array, mean_spectrum, color='steelblue', linewidth=0.8, label='Mean spectrum')

	if peak_mz is not None:
		peak_intensities = np.interp(peak_mz, mz_array, mean_spectrum)
		ax.vlines(peak_mz, 0, peak_intensities, colors='tomato',
				  linewidth=0.6, alpha=0.7, label='Detected peaks')

	ax.set_xlabel('m/z')
	ax.set_ylabel('Normalised intensity (TIC)')
	ax.set_title(title)
	ax.legend(fontsize=9)
	plt.tight_layout()
	return fig

########################################################################################

def plot_ion_images(peak_matrix, peak_mz, mz_list, coordinates, image_shape, cmap='inferno'):
	"""
	Plot ion images for a list of target m/z values in a grid.

	Parameters:
	peak_matrix (np.ndarray): Aligned intensity matrix (n_spectra, n_peaks).
	peak_mz (np.ndarray): 1D array of aligned peak m/z values.
	mz_list (list of float): Target m/z values to visualise.
	coordinates (list of tuple): (x, y) pixel coordinates.
	image_shape (tuple): (height, width) of the tissue image.
	cmap (str): Matplotlib colormap name. Default: 'inferno'.

	Returns:
	matplotlib.figure.Figure
	"""
	n = len(mz_list)
	ncols = min(n, 4)
	nrows = int(np.ceil(n / ncols))

	fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
	axes = np.array(axes).ravel()

	for i, target_mz in enumerate(mz_list):
		ion_img = generate_ion_image(peak_matrix, peak_mz, target_mz, coordinates, image_shape)
		ax = axes[i]
		im = ax.imshow(ion_img, cmap=cmap, aspect='equal')
		divider = make_axes_locatable(ax)
		cax_cb = divider.append_axes("right", size="5%", pad=0.1)
		fig.colorbar(im, cax=cax_cb)
		ax.set_title(f'm/z {target_mz:.2f}', fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])

	for ax in axes[n:]:
		fig.delaxes(ax)

	plt.tight_layout()
	return fig

########################################################################################

def plot_pca_results(pc_images, loadings_df, explained_variance, n_components_to_show=3):
	"""
	Visualise PCA results: spatial maps for the first n PCs and a scree plot.

	Parameters:
	pc_images (list of np.ndarray): Spatial PC maps from run_pca().
	loadings_df (pd.DataFrame): Loadings DataFrame from run_pca().
	explained_variance (np.ndarray): Fraction of variance explained per PC.
	n_components_to_show (int): Number of PC spatial maps to display. Default: 3.

	Returns:
	matplotlib.figure.Figure
	"""
	n = min(n_components_to_show, len(pc_images))
	fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

	for i in range(n):
		ax = axes[i]
		im = ax.imshow(pc_images[i], cmap='RdBu_r', aspect='equal')
		divider = make_axes_locatable(ax)
		cax_cb = divider.append_axes("right", size="5%", pad=0.1)
		fig.colorbar(im, cax=cax_cb)
		ax.set_title(f'PC{i + 1} ({explained_variance[i] * 100:.1f}%)', fontsize=11)
		ax.set_xticks([])
		ax.set_yticks([])

	ax = axes[-1]
	cumvar = np.cumsum(explained_variance) * 100
	ax.bar(range(1, len(explained_variance) + 1), explained_variance * 100, color='steelblue')
	ax.plot(range(1, len(explained_variance) + 1), cumvar, 'o-', color='tomato', label='Cumulative')
	ax.set_xlabel('Principal Component')
	ax.set_ylabel('Explained variance (%)')
	ax.set_title('Scree plot')
	ax.legend()

	plt.tight_layout()
	return fig

########################################################################################

def plot_segmentation(segment_image, n_clusters, title='Bisecting k-means segmentation'):
	"""
	Visualise the bisecting k-means segmentation map.

	Parameters:
	segment_image (np.ndarray): 2D integer array from run_bisecting_kmeans().
	n_clusters (int): Number of clusters.
	title (str): Figure title.

	Returns:
	matplotlib.figure.Figure
	"""
	colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
	cmap = ListedColormap(colors)

	display = segment_image.astype(float)
	display[segment_image == -1] = np.nan

	fig, ax = plt.subplots(figsize=(7, 7))
	im = ax.imshow(display, cmap=cmap, vmin=0, vmax=n_clusters - 1, aspect='equal')
	cbar = fig.colorbar(im, ax=ax, ticks=range(n_clusters))
	cbar.set_ticklabels([f'Cluster {k + 1}' for k in range(n_clusters)])
	ax.set_title(title, fontsize=12)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.tight_layout()
	return fig

########################################################################################

def plot_roc_summary(roc_df, auc_high=0.6, auc_low=0.4):
	"""
	Scatter plot of ROC AUC vs m/z, highlighting discriminative peaks.

	Parameters:
	roc_df (pd.DataFrame): ROC results from compute_roc_per_mz().
	auc_high (float): Upper AUC threshold. Default: 0.6.
	auc_low (float): Lower AUC threshold. Default: 0.4.

	Returns:
	matplotlib.figure.Figure
	"""
	disc = (roc_df['auc'] >= auc_high) | (roc_df['auc'] <= auc_low)

	fig, ax = plt.subplots(figsize=(12, 5))
	ax.scatter(roc_df.loc[~disc, 'mz'], roc_df.loc[~disc, 'auc'],
			   color='lightgray', s=10, label='Non-discriminative', zorder=2)
	ax.scatter(roc_df.loc[disc, 'mz'], roc_df.loc[disc, 'auc'],
			   color='tomato', s=15,
			   label=f'Discriminative (AUC \u2265{auc_high} or \u2264{auc_low})', zorder=3)
	ax.axhline(auc_high, color='tomato', linestyle='--', linewidth=0.8)
	ax.axhline(auc_low, color='tomato', linestyle='--', linewidth=0.8)
	ax.axhline(0.5, color='gray', linestyle='-', linewidth=0.5)
	ax.set_xlabel('m/z')
	ax.set_ylabel('AUC')
	ax.set_title('ROC AUC per m/z')
	ax.set_ylim(0, 1)
	ax.legend(fontsize=9)
	plt.tight_layout()
	return fig