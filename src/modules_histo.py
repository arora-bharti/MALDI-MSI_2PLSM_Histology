#!/usr/bin/env python
# coding: utf-8
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

# This file contains all the modules/functions necessary for running the jupyter notebook.

########################################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # must be set before TensorFlow/StarDist are imported

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from PIL import Image
from tqdm.auto import tqdm
from skimage.measure import regionprops
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
from stardist import random_label_cmap
from csbdeep.data import Normalizer, normalize_mi_ma

Image.MAX_IMAGE_PIXELS = None  # must be set before any image is opened

##########################################################################

def weighted_kde_density_map(nucleus_mask, bandwidth = 'auto', kernel = 'gaussian', num_points = 1000):
	"""
	Compute the weighted kernel density estimate (KDE) of the centroids of regions in a binary image.

	Parameters:
		nucleus_mask (ndarray): Binary image of nuclei, with 1's indicating nuclei and 0's indicating background.
		bandwidth (float or str, optional): The bandwidth of the KDE. 
		If 'auto', use the rule of thumb bandwidth. Default is 'auto'.
		kernel (str, optional): The kernel function to use. Default is 'gaussian'.
		num_points (int, optional): The number of points to use in the density map. Default is 500.

	Returns:
		density_map (ndarray): The density map of the centroids of the nuclei.
	"""
	
	# Extract centroid locations and areas of each nucleus
	regions = regionprops(nucleus_mask)
	nucleus_centroids = np.array([region.centroid for region in regions])
	nucleus_areas = np.array([region.area for region in regions])

	# Compute the weighted KDE
	if bandwidth == 'auto':
		# Use the rule of thumb bandwidth selection
		num_nuclei = nucleus_centroids.shape[0]
		num_dimensions = nucleus_centroids.shape[1]
		bandwidth = num_nuclei**(-1.0/(num_dimensions+4)) * np.std(nucleus_centroids, axis=0).mean()

		# make bandwidth more refined than the calculated amount.
		bandwidth = float(0.5 * bandwidth)

	else:
		bandwidth = float(bandwidth)  # Ensure bandwidth is a float

	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(nucleus_centroids, sample_weight=nucleus_areas)

	# Compute the size of the grid for the density map based on the desired number of points
	num_steps_x = int(np.ceil(np.sqrt(num_points * nucleus_mask.shape[1] / nucleus_mask.shape[0])))
	num_steps_y = int(np.ceil(np.sqrt(num_points * nucleus_mask.shape[0] / nucleus_mask.shape[1])))
	x_step_size = int(np.ceil(nucleus_mask.shape[1] / num_steps_x))
	y_step_size = int(np.ceil(nucleus_mask.shape[0] / num_steps_y))

	# Create a density map
	x_steps = int(np.ceil(nucleus_mask.shape[1] / x_step_size))
	y_steps = int(np.ceil(nucleus_mask.shape[0] / y_step_size))
	x_coords = np.arange(0, x_steps * x_step_size, x_step_size)
	y_coords = np.arange(0, y_steps * y_step_size, y_step_size)
	xx, yy = np.meshgrid(x_coords, y_coords)

	# Evaluate the KDE at each point in the grid to create the density map
	grid_points = np.vstack([yy.ravel(), xx.ravel()]).T
	density = np.exp(kde.score_samples(grid_points))
	density_map = density.reshape((y_steps, x_steps))
	
	# Smoothing parameter (adjust as needed)
	smoothing_sigma = 2

	# Apply Gaussian smoothing to the density map
	smoothed_density_map = gaussian_filter(density_map, sigma=smoothing_sigma)

	return smoothed_density_map

##########################################################################


def normalize_array(arr):
            # Find the 1st and 99th percentiles
            p1, p99 = np.percentile(arr, [1, 99])

            # Subtract the 1st percentile from all elements
            arr_normalized = arr - p1

            # Divide all elements by the difference between the 99th and 1st percentiles
            arr_normalized /= (p99 - p1)

            # Multiply all elements by 255
            arr_normalized *= 255

            return arr_normalized


##########################################################################


def mean_filter(labels, size = 0.1):
	"""
	Calculates local density of an input array.
	# Compute label areas
	label_areas = np.bincount(label_image.flat)

	Parameters:
		labels (numpy array): 2D array of labels
	# Create kernel density estimate of centroids, weighted by label areas
	weights = label_areas[label_image] / np.sum(label_areas)
	kde = gaussian_kde(centroids.T, weights=weights)

	Returns:
		Local_Density (numpy array): 2D array of local density values
	# Define grid for heatmap
	shape = label_image.shape
	x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
	grid_coords = np.vstack([x.ravel(), y.ravel()])

	"""
	
	binary_image = np.where(labels > 0, 255, 0).astype(np.uint8)

	# Calculate window size as 10% of the minimum dimension of the input array
	window_size = int(0.1 * min(binary_image.shape[0], binary_image.shape[1]))

	# Ensure that window_size is a positive integer
	window_size = max(1, window_size)

	# Apply a mean filter to the grayscale image using the calculated window size
	Local_Density = cv2.blur(binary_image, (window_size, window_size))

	return Local_Density

##########################################################################

def normalize_density_maps(Local_Density):

	Normalized_Local_Density = np.divide(Local_Density, Local_Density.max(), 
										out=np.full(Local_Density.shape, np.nan), 
										where=Local_Density.max() != 0)

	return Normalized_Local_Density

##########################################################################

def show_image(img, **kwargs):
	"""Plot the original image."""
	plt.imshow(img, **kwargs)
	plt.axis('off')
	plt.show()

########################################################################################


def make_binned_image(label_image, label_properties, Parameter, num_bins):
	
	dataframe = pd.DataFrame(label_properties)
	
	# Get the minimum and maximum parameter values
	min_parameter = max(np.min(dataframe[Parameter]), 1e-2)
	max_parameter = np.max(dataframe[Parameter])

	# Define the parameter bins using np.linspace
	parameter_bins = np.linspace(min_parameter, max_parameter, num_bins)  # Create 'num_bins' bins

	# Define the corresponding labels for the bins based on the number of bins
	parameter_labels = list(range(1, num_bins + 1))

	# Initialize an empty labeled image with the same shape as label_image
	binned_label_image = np.full_like(label_image, 0, dtype=float)  # Initialize with zeros

	# Create a dictionary to keep track of the count of components in each bin
	bin_counts = {label: 0 for label in parameter_labels}

	# Calculate parameter values of connected components and map them to bins
	for region in tqdm(regionprops(label_image)):
		parameter_value = getattr(region, Parameter)  # Use getattr() to access the attribute
		for i, bin_edge in enumerate(parameter_bins[1:]):
			if parameter_value <= bin_edge:
				binned_label_image[label_image == region.label] = parameter_labels[i]
				bin_counts[parameter_labels[i]] += 1
				break

	return binned_label_image

########################################################################################


# Use the percentile for normalization
class MyNormalizer(Normalizer):
	def __init__(self, mi, ma):
			self.mi, self.ma = mi, ma
	def before(self, x, axes):
		return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)
	def after(*args, **kwargs):
		assert False
	@property
	def do_after(self):
		return False

########################################################################################

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

def make_mosaic_and_save(img, labels, num_clusters, Local_Density_mean_filter, Local_Packing, eccentricity_binned_image, area_binned_image, filename, output_folder):
	
	random_cmap = random_label_cmap()

	fig, axs = plt.subplots(2, 3, figsize=(36, 24))

	# Define the custom colormap
	colors = ['black', 'red', 'green', 'yellow', 'orange', 'cyan', 'lime']
	custom_cmap = ListedColormap(colors)

	# Normalize images for display
	images = [img, labels, Local_Density_mean_filter, Local_Packing, eccentricity_binned_image, area_binned_image]
	images = [(im - im.min()) / (im.max() - im.min()) for im in images]
	
	titles = ['Image', 'Label', 'Distribution', 'Density', 'Eccentricity', 'Area']
	cmaps = [None, random_cmap, 'inferno', 'inferno', custom_cmap, custom_cmap]  
	ticks = [None, None, np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3)]  
	
	for ax, im, title, cmap, tick in zip(axs.flat, images, titles, cmaps, ticks):
		im_display = ax.imshow(im, cmap=cmap)
		ax.set_title(title, fontsize=26, pad = 20)  
		ax.axis('off')  
		
		if tick is not None:
			divider = make_axes_locatable(ax)
			cax = divider.append_axes('right', size='5%', pad=0.07)
			cbar = fig.colorbar(im_display, cax=cax, ticks=tick)
			cbar.ax.tick_params(labelsize=20)
			
			# Remove tick labels for 'Eccentricity' and 'Area' subplots
			if title in ['Eccentricity', 'Area']:
				cbar.ax.set_yticklabels([])
	
	# Save the mosaic as .png in the results folder
#     plt.subplots_adjust(wspace=0.4, hspace=0.4)
	output_path = os.path.join(output_folder, filename.replace('.tif', '_mosaic.png'))
	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.close()

	print(f"Mosaic saved to {output_path}")
	print()

########################################################################################

def normalize_staining(image, target_concentrations=None):
	"""
	Normalise H&E staining using colour deconvolution (Ruifrok & Johnston method).

	Converts the RGB image to optical density (OD) space using known H&E stain
	vectors, deconvolves the stain concentrations, and reconstructs a normalised
	RGB image scaled to a reference maximum concentration. This corrects for
	batch-to-batch staining variability across slides and scanners.

	No additional dependencies required — uses scikit-image, already in requirements.

	Parameters:
	image (np.ndarray): RGB H&E image, dtype uint8, shape (H, W, 3).
	target_concentrations (np.ndarray or None): Reference maximum concentration
		for each stain [H_max, E_max]. Default: [1.9, 1.0] from typical H&E slides.

	Returns:
	np.ndarray: Normalised RGB image, uint8, same shape as input.
	"""
	from skimage.color import separate_stains, combine_stains

	# Standard H&E stain matrix (Ruifrok & Johnston, Anal Quant Cytol Histol 2001)
	HE_stain_matrix = np.array([
		[0.65, 0.70, 0.29],
		[0.07, 0.99, 0.11],
		[0.27, 0.57, 0.78]
	])

	if target_concentrations is None:
		target_concentrations = np.array([1.9, 1.0])

	# Deconvolve: separate_stains returns OD concentrations per stain channel
	stains = separate_stains(image.astype(np.float32) / 255.0, HE_stain_matrix)

	# Normalise each stain channel to its target maximum
	for ch in range(2):
		ch_max = stains[:, :, ch].max()
		if ch_max > 0:
			stains[:, :, ch] = stains[:, :, ch] / ch_max * target_concentrations[ch]

	# Zero out the third (background) channel
	stains[:, :, 2] = 0.0

	# Reconstruct RGB
	normalised = combine_stains(stains, np.linalg.inv(HE_stain_matrix))
	normalised = np.clip(normalised * 255.0, 0, 255).astype(np.uint8)

	return normalised

########################################################################################

def make_nuclei_network(label_image, max_distance=None):
	"""
	Build a spatial proximity network of detected nuclei.

	Each nucleus centroid is a node. An edge connects two nuclei if their
	centroids are within max_distance pixels of each other. Edge weight is
	the inverse of the Euclidean distance (closer nuclei → stronger connection).

	Requires: pip install networkx

	Parameters:
	label_image (np.ndarray): Integer label image from StarDist (each nucleus
		has a unique positive integer label; background is 0).
	max_distance (float or None): Maximum centroid-to-centroid distance in pixels
		for edge creation. If None, uses 10% of the shorter image dimension.

	Returns:
	tuple:
		G (networkx.Graph): Spatial nuclei network. Node attributes: 'centroid',
			'area'. Edge attributes: 'weight' (inverse distance).
		centroids (np.ndarray): Array of shape (n_nuclei, 2) with (row, col)
			centroid coordinates.
	"""
	try:
		import networkx as nx
	except ImportError:
		raise ImportError("networkx is required. Install with: pip install networkx")

	if max_distance is None:
		max_distance = 0.10 * min(label_image.shape)

	regions = regionprops(label_image)
	centroids = np.array([r.centroid for r in regions])
	areas = np.array([r.area for r in regions])
	labels = np.array([r.label for r in regions])

	G = nx.Graph()
	for i, (centroid, area, lbl) in enumerate(zip(centroids, areas, labels)):
		G.add_node(i, centroid=tuple(centroid), area=int(area), label=int(lbl))

	n = len(centroids)
	for i in range(n):
		for j in range(i + 1, n):
			dist = float(np.linalg.norm(centroids[i] - centroids[j]))
			if dist <= max_distance and dist > 0:
				G.add_edge(i, j, weight=1.0 / dist)

	return G, centroids

########################################################################################

def make_voronoi_tessellation(label_image):
	"""
	Compute the Voronoi tessellation of nucleus centroids.

	Each nucleus centroid becomes a Voronoi site. The tessellation partitions
	the image plane into regions of influence for each nucleus — larger Voronoi
	regions indicate more isolated nuclei (lower local density).

	Uses scipy.spatial.Voronoi, already a project dependency.

	Parameters:
	label_image (np.ndarray): Integer label image from StarDist segmentation.

	Returns:
	tuple:
		vor (scipy.spatial.Voronoi): Fitted Voronoi object.
		centroids (np.ndarray): Array of shape (n_nuclei, 2) with (col, row)
			coordinates in (x, y) order as required by scipy.spatial.Voronoi.
	"""
	from scipy.spatial import Voronoi

	regions = regionprops(label_image)
	# Voronoi expects (x, y) = (col, row)
	centroids = np.array([[r.centroid[1], r.centroid[0]] for r in regions])

	if len(centroids) < 4:
		raise ValueError("Voronoi tessellation requires at least 4 nuclei.")

	vor = Voronoi(centroids)
	return vor, centroids

########################################################################################

def plot_nuclei_network(label_image, G, centroids, image=None):
	"""
	Visualise the nuclei spatial proximity network overlaid on the tissue image.

	Parameters:
	label_image (np.ndarray): Integer label image from StarDist segmentation.
	G (networkx.Graph): Output of make_nuclei_network().
	centroids (np.ndarray): Centroid array (n_nuclei, 2) in (row, col) order.
	image (np.ndarray or None): Background RGB or grayscale image. Optional.

	Returns:
	matplotlib.figure.Figure
	"""
	fig, ax = plt.subplots(figsize=(10, 10))

	background = image if image is not None else label_image
	ax.imshow(background, cmap='gray' if background.ndim == 2 else None, alpha=0.6)

	for u, v in G.edges():
		y0, x0 = centroids[u]
		y1, x1 = centroids[v]
		ax.plot([x0, x1], [y0, y1], color='steelblue', linewidth=0.4, alpha=0.5)

	ax.scatter(centroids[:, 1], centroids[:, 0],
			   s=6, color='tomato', zorder=3,
			   label=f'Nuclei (n={len(centroids)})')

	ax.set_title(
		f'Nuclei network  |  nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}',
		fontsize=12)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.legend(fontsize=9, loc='upper right')
	plt.tight_layout()
	return fig

########################################################################################

def plot_voronoi(vor, centroids, image_shape, image=None):
	"""
	Visualise the Voronoi tessellation of nucleus centroids.

	Parameters:
	vor (scipy.spatial.Voronoi): Output of make_voronoi_tessellation().
	centroids (np.ndarray): Centroid array (n_nuclei, 2) in (col, row) / (x, y) order.
	image_shape (tuple): (height, width) of the tissue image.
	image (np.ndarray or None): Background image to display. Optional.

	Returns:
	matplotlib.figure.Figure
	"""
	from scipy.spatial import voronoi_plot_2d

	fig, ax = plt.subplots(figsize=(10, 10))

	if image is not None:
		ax.imshow(image, cmap='gray' if image.ndim == 2 else None, alpha=0.5,
				  extent=[0, image_shape[1], image_shape[0], 0])

	voronoi_plot_2d(vor, ax=ax, show_vertices=False,
					line_colors='steelblue', line_width=0.6, point_size=3)

	ax.set_xlim(0, image_shape[1])
	ax.set_ylim(image_shape[0], 0)
	ax.set_title('Voronoi tessellation of nucleus centroids', fontsize=12)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.tight_layout()
	return fig