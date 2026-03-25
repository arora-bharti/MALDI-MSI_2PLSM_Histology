# Scripts

This folder contains utility scripts for the analysis pipeline.

## Files

### Merging.ijm
ImageJ/Fiji macro for post-processing and merging analysis outputs.

**Workflow:**
1. Opens original image and corresponding density-filtered image
2. Applies 16-color reduction to density image
3. Creates Z-stack from both images
4. Generates maximum intensity projection
5. Saves merged result as PNG

**Expected Directory Structure:**
```
project/
├── input_data/          # Original images
├── output_data/
│   └── density/         # Density-filtered images
└── merged/              # Output merged images
```

**Usage:**
1. Open ImageJ/Fiji
2. Plugins → Macros → Run...
3. Select Merging.ijm
4. Adjust input/output paths in the script as needed
