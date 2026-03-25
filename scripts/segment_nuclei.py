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
Nuclei Segmentation Script

Segments nuclei from H&E histology images using StarDist.

Usage:
    python scripts/segment_nuclei.py --input path/to/images --output path/to/results
    python scripts/segment_nuclei.py --input histology.tif --output results/
    python scripts/segment_nuclei.py --input data/ --output results/ --prob-thresh 0.5
"""

import argparse
import os
import sys
import warnings
from contextlib import redirect_stdout

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from tqdm import tqdm
from skimage.measure import regionprops_table

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules_histo import (
    MyNormalizer,
    mean_filter,
    normalize_density_maps,
    weighted_kde_density_map,
    make_binned_image,
)


def segment_single_image(image_path, output_dir, model, prob_thresh, nms_thresh,
                         norm_percentiles, num_bins, compute_morphometry=True):
    """
    Segment nuclei from a single H&E image.

    Parameters:
        image_path: Path to input image (TIFF, RGB)
        output_dir: Directory to save results
        model: StarDist2D model
        prob_thresh: Detection probability threshold
        nms_thresh: Non-maximum suppression threshold
        norm_percentiles: Tuple of (low, high) percentiles for normalization
        num_bins: Number of bins for morphometry
        compute_morphometry: Whether to compute eccentricity/area binning

    Returns:
        dict: Segmentation results
    """
    from stardist.utils import fill_label_holes
    import cv2

    # Load image
    img = imread(image_path)

    # Setup normalizer
    mi, ma = np.percentile(img, norm_percentiles)
    normalizer = MyNormalizer(mi, ma)

    # Configure model
    model.config.prob_thresh = prob_thresh
    model.config.nms_thresh = nms_thresh

    # Predict
    with redirect_stdout(open(os.devnull, "w")):
        if img.ndim == 3:  # RGB image
            labels, polys = model.predict_instances_big(
                img, axes='YXC', block_size=1000,
                min_overlap=144, context=128,
                normalizer=normalizer, n_tiles=(2, 2, 1),
                show_progress=False
            )
        else:  # Grayscale
            labels, polys = model.predict_instances(img, normalizer=normalizer)

    labels = fill_label_holes(labels)

    # Get filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save labeled image
    labeled_dir = os.path.join(output_dir, 'labeled')
    os.makedirs(labeled_dir, exist_ok=True)
    imwrite(os.path.join(labeled_dir, f"{base_name}_labels.tif"), labels.astype(np.uint16))

    # Calculate density
    local_density = mean_filter(labels)
    local_density = normalize_density_maps(local_density)

    density_dir = os.path.join(output_dir, 'density')
    os.makedirs(density_dir, exist_ok=True)
    imwrite(os.path.join(density_dir, f"{base_name}_density.tif"), local_density.astype(np.float32))

    # Calculate statistics
    # Convert to grayscale for thresholding
    if img.ndim == 3:
        image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = img

    threshold_value = 200
    accepted_mask = image_gray < threshold_value

    density_high = local_density >= 0.5
    density_low = local_density < 0.5

    density_high_accepted = density_high & accepted_mask
    density_low_accepted = density_low & accepted_mask

    accepted_count = np.sum(accepted_mask)
    pct_high = (np.sum(density_high_accepted) / accepted_count) * 100 if accepted_count > 0 else 0
    pct_low = (np.sum(density_low_accepted) / accepted_count) * 100 if accepted_count > 0 else 0

    # Nuclear packaging (KDE)
    try:
        local_packing = weighted_kde_density_map(labels, num_points=5000)
        local_packing = normalize_density_maps(local_packing)

        packaging_dir = os.path.join(output_dir, 'packaging')
        os.makedirs(packaging_dir, exist_ok=True)
        imwrite(os.path.join(packaging_dir, f"{base_name}_packaging.tif"),
                local_packing.astype(np.float32))
    except Exception:
        pass  # Skip if KDE fails (e.g., too few nuclei)

    # Morphometry (optional)
    if compute_morphometry and np.max(labels) > 0:
        label_properties = regionprops_table(labels, intensity_image=img,
                                             properties=('label', 'area', 'eccentricity'))

        # Eccentricity binning
        try:
            ecc_binned = make_binned_image(labels, label_properties, 'eccentricity', num_bins)
            ecc_dir = os.path.join(output_dir, 'eccentricity')
            os.makedirs(ecc_dir, exist_ok=True)
            imwrite(os.path.join(ecc_dir, f"{base_name}_eccentricity.tif"), ecc_binned.astype(np.float32))
        except Exception:
            pass

        # Area binning
        try:
            area_binned = make_binned_image(labels, label_properties, 'area', num_bins)
            area_dir = os.path.join(output_dir, 'area')
            os.makedirs(area_dir, exist_ok=True)
            imwrite(os.path.join(area_dir, f"{base_name}_area.tif"), area_binned.astype(np.float32))
        except Exception:
            pass

    return {
        'filename': os.path.basename(image_path),
        'num_nuclei': int(np.max(labels)),
        'pct_high_density': round(pct_high, 2),
        'pct_low_density': round(pct_low, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Segment nuclei from H&E histology images using StarDist',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Segment single image
  python scripts/segment_nuclei.py -i histology.tif -o results/

  # Segment folder of images
  python scripts/segment_nuclei.py -i data/histology/ -o results/

  # Custom thresholds
  python scripts/segment_nuclei.py -i data/ -o results/ --prob-thresh 0.5 --nms-thresh 0.4
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Input image or directory containing .tif images')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for results')
    parser.add_argument('--prob-thresh', type=float, default=0.4,
                        help='Detection probability threshold (default: 0.4)')
    parser.add_argument('--nms-thresh', type=float, default=0.3,
                        help='Non-maximum suppression threshold (default: 0.3)')
    parser.add_argument('--norm-low', type=float, default=25.0,
                        help='Low percentile for normalization (default: 25.0)')
    parser.add_argument('--norm-high', type=float, default=85.0,
                        help='High percentile for normalization (default: 85.0)')
    parser.add_argument('--num-bins', type=int, default=6,
                        help='Number of bins for morphometry (default: 6)')
    parser.add_argument('--skip-morphometry', action='store_true',
                        help='Skip eccentricity/area binning')

    args = parser.parse_args()

    # Import StarDist (heavy import, do it after arg parsing)
    print("Loading StarDist model...")
    from stardist.models import StarDist2D
    model = StarDist2D.from_pretrained('2D_versatile_he')
    print("Model loaded.")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get list of images
    if os.path.isfile(args.input):
        image_files = [args.input]
    else:
        import glob
        image_files = sorted(glob.glob(os.path.join(args.input, '*.tif')))

    if not image_files:
        print(f"No .tif files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) to process")
    print(f"Parameters: prob_thresh={args.prob_thresh}, nms_thresh={args.nms_thresh}")
    print()

    # Process images
    all_results = []
    for image_path in tqdm(image_files, desc='Segmenting'):
        try:
            result = segment_single_image(
                image_path,
                args.output,
                model,
                args.prob_thresh,
                args.nms_thresh,
                (args.norm_low, args.norm_high),
                args.num_bins,
                compute_morphometry=not args.skip_morphometry
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # Save summary CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(args.output, 'segmentation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
