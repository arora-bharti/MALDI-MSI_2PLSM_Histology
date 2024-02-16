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

import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tifffile import imread, imsave
from csbdeep.utils import Path, normalize

from stardist import random_label_cmap
from stardist.models import StarDist2D

import pandas as pd
from csbdeep.data import Normalizer, normalize_mi_ma

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from csbdeep.utils import Path, normalize
from csbdeep.data import Normalizer, normalize_mi_ma

from tqdm.auto import tqdm

from skimage.measure import regionprops
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter

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