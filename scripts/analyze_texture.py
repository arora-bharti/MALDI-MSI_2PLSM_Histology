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
Collagen Texture Analysis Script

Analyzes 2PLSM images for collagen fiber coherence, orientation, and local density.

Usage:
    python scripts/analyze_texture.py --input path/to/images --output path/to/results
    python scripts/analyze_texture.py --input image.tif --output results/
    python scripts/analyze_texture.py --input data/ --output results/ --filter-sigma 3 --local-sigma 15
"""

import argparse
import os
import sys
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules_2photon import (
    make_filtered_image,
    binarize_image,
    convolve,
    make_image_gradients,
    make_structure_tensor_2d,
    make_coherence,
    make_orientation,
    make_vxvy,
    perform_statistical_analysis,
)


def analyze_single_image(image_path, output_dir, filter_sigma, local_density_kernel,
                         local_sigma, save_intermediates=True):
    """
    Analyze a single 2PLSM image for texture features.

    Parameters:
        image_path: Path to input image (TIFF)
        output_dir: Directory to save results
        filter_sigma: Gaussian filter sigma
        local_density_kernel: Kernel size for local density
        local_sigma: Structure tensor local sigma
        save_intermediates: Whether to save intermediate images

    Returns:
        dict: Analysis results
    """
    # Load and normalize image
    raw_image = np.array(Image.open(image_path).convert("L"))
    raw_image = 255 * ((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min() + 1e-8))

    # Calculate threshold
    threshold = max(int(np.median(raw_image)), 2)

    # Filter image
    filtered_image = make_filtered_image(raw_image, filter_sigma)

    # Binarize
    binarized_image = binarize_image(filtered_image)

    # Local density
    kernel_size = local_density_kernel if local_density_kernel % 2 == 1 else local_density_kernel + 1
    kernel_size = max(kernel_size, 3)
    local_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    local_density = convolve(raw_image, local_kernel)
    local_density = np.divide(local_density, local_density.max(),
                              out=np.full(local_density.shape, np.nan),
                              where=local_density.max() != 0)

    # Structure tensor analysis
    grad_x, grad_y = make_image_gradients(filtered_image)
    structure_tensor, eigenvalues, eigenvectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(
        grad_x, grad_y, local_sigma
    )

    # Coherence and orientation
    coherence = make_coherence(filtered_image, eigenvalues, structure_tensor, threshold)
    orientation = make_orientation(filtered_image, Jxx, Jxy, Jyy, threshold)
    vx, vy = make_vxvy(filtered_image, eigenvectors, threshold)

    # Statistics
    results = perform_statistical_analysis(os.path.basename(image_path), local_sigma, coherence)

    # Get filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save outputs
    if save_intermediates:
        Image.fromarray(filtered_image.astype(np.float32)).save(
            os.path.join(output_dir, f"{base_name}_filtered.tif")
        )
        Image.fromarray(local_density.astype(np.float32)).save(
            os.path.join(output_dir, f"{base_name}_density.tif")
        )

    Image.fromarray(coherence.astype(np.float32)).save(
        os.path.join(output_dir, f"{base_name}_coherence.tif")
    )
    Image.fromarray(orientation.astype(np.float32)).save(
        os.path.join(output_dir, f"{base_name}_orientation.tif")
    )

    return {
        'filename': os.path.basename(image_path),
        'low_coherence_pct': results[0, 1],
        'high_coherence_pct': results[0, 2],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze 2PLSM images for collagen texture features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  python scripts/analyze_texture.py -i image.tif -o results/

  # Analyze folder of images
  python scripts/analyze_texture.py -i data/images/ -o results/

  # Custom parameters
  python scripts/analyze_texture.py -i data/ -o results/ --filter-sigma 3 --local-sigma 15
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Input image or directory containing .tif images')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for results')
    parser.add_argument('--filter-sigma', type=float, default=2.0,
                        help='Gaussian filter sigma (default: 2.0)')
    parser.add_argument('--local-density-kernel', type=int, default=20,
                        help='Kernel size for local density (default: 20)')
    parser.add_argument('--local-sigma', type=float, default=10.0,
                        help='Structure tensor local sigma (default: 10.0)')
    parser.add_argument('--no-intermediates', action='store_true',
                        help='Skip saving intermediate images (filtered, density)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get list of images
    if os.path.isfile(args.input):
        image_files = [args.input]
    else:
        image_files = sorted(glob.glob(os.path.join(args.input, '*.tif')))
        # Exclude already processed files
        exclude_prefixes = ['FilteredImage_', 'DensityImage_', 'CoheranceImage_',
                           'OrientationImage_', 'Results_']
        image_files = [f for f in image_files
                       if not any(os.path.basename(f).startswith(p) for p in exclude_prefixes)]

    if not image_files:
        print(f"No .tif files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) to process")
    print(f"Parameters: filter_sigma={args.filter_sigma}, local_sigma={args.local_sigma}")
    print()

    # Process images
    all_results = []
    for image_path in tqdm(image_files, desc='Processing'):
        try:
            result = analyze_single_image(
                image_path,
                args.output,
                args.filter_sigma,
                args.local_density_kernel,
                args.local_sigma,
                save_intermediates=not args.no_intermediates
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # Save summary CSV
    if all_results:
        import pandas as pd
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(args.output, 'texture_analysis_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
