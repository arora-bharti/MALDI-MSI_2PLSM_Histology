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
"""

from .modules_2photon import (
    binarize_image,
    make_filtered_image,
    percentage_area,
    make_image_gradients,
    make_structure_tensor_2d,
    make_coherence,
    make_orientation,
    make_vxvy,
    convolve,
    make_mosiac_plot,
    perform_statistical_analysis,
    convert_to_8bit_grayscale,
)

from .modules_histo import (
    weighted_kde_density_map,
    normalize_array,
    mean_filter,
    normalize_density_maps,
    make_binned_image,
    MyNormalizer,
    make_mosaic_and_save,
)

from .modules_maldi import (
    read_imzml,
    normalize_tic,
    find_and_align_peaks,
    generate_ion_image,
    build_data_cube,
    import_qupath_annotations,
    compute_roc_per_mz,
    get_discriminative_mz,
    run_pca,
    run_bisecting_kmeans,
    plot_mean_spectrum,
    plot_ion_images,
    plot_pca_results,
    plot_segmentation,
    plot_roc_summary,
)

__version__ = "1.0.0"
__author__ = "Bharti Arora, Ajinkya Kulkarni"
__email__ = "bharti.arora@mpinat.mpg.de"
