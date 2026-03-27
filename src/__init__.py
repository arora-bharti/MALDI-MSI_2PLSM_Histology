#!/usr/bin/env python3
# encoding: utf-8
#
# Copyright (C) 2024 Max Planck Institute for Multidisclplinary Sciences
# Copyright (C) 2024 Bharti Arora <bharti.arora@mpinat.mpg.de>
# Copyright (C) 2024 Ajinkya Kulkarni <ajinkya.kulkarni@mpinat.mpg.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""
MALDI-MSI_2PLSM_Histology Analysis Package

This package provides modules for:
- 2-photon microscopy texture analysis (collagen coherence, orientation)
- Histology/nuclei segmentation and density analysis
- MALDI-MSI analysis (open-source SCiLS Lab replacement)
"""

# Explicit re-exports (`x as x` marks these as intentional public API)
from .modules_2photon import (
    binarize_image as binarize_image,
    make_filtered_image as make_filtered_image,
    percentage_area as percentage_area,
    make_image_gradients as make_image_gradients,
    make_structure_tensor_2d as make_structure_tensor_2d,
    make_coherence as make_coherence,
    make_orientation as make_orientation,
    make_vxvy as make_vxvy,
    convolve as convolve,
    make_mosiac_plot as make_mosiac_plot,
    perform_statistical_analysis as perform_statistical_analysis,
    load_pandas_dataframe as load_pandas_dataframe,
    convert_to_8bit_grayscale as convert_to_8bit_grayscale,
)

from .modules_histo import (
    weighted_kde_density_map as weighted_kde_density_map,
    normalize_array as normalize_array,
    mean_filter as mean_filter,
    normalize_density_maps as normalize_density_maps,
    make_binned_image as make_binned_image,
    MyNormalizer as MyNormalizer,
    make_mosaic_and_save as make_mosaic_and_save,
    normalize_staining as normalize_staining,
    make_nuclei_network as make_nuclei_network,
    make_voronoi_tessellation as make_voronoi_tessellation,
    plot_nuclei_network as plot_nuclei_network,
    plot_voronoi as plot_voronoi,
)

from .modules_maldi import (
    read_imzml as read_imzml,
    normalize_tic as normalize_tic,
    find_and_align_peaks as find_and_align_peaks,
    generate_ion_image as generate_ion_image,
    build_data_cube as build_data_cube,
    import_qupath_annotations as import_qupath_annotations,
    compute_roc_per_mz as compute_roc_per_mz,
    get_discriminative_mz as get_discriminative_mz,
    run_pca as run_pca,
    run_bisecting_kmeans as run_bisecting_kmeans,
    plot_mean_spectrum as plot_mean_spectrum,
    plot_ion_images as plot_ion_images,
    plot_pca_results as plot_pca_results,
    plot_segmentation as plot_segmentation,
    plot_roc_summary as plot_roc_summary,
)

__version__ = "1.0.0"
__author__ = "Bharti Arora, Ajinkya Kulkarni"
__email__ = "bharti.arora@mpinat.mpg.de"
