// Merging.ijm
// Overlay a density map onto its corresponding original image and save as PNG.
//
// Usage: Open the original image in Fiji first, then run this macro.
// A dialog will ask for the density images folder and the output folder.

// ── User-configurable paths (via dialog) ────────────────────────────────────────
Dialog.create("Merging: Set Folders");
Dialog.addDirectory("Density images folder", "");
Dialog.addDirectory("Output (merged) folder", "");
Dialog.show();

String baseOutputPath   = Dialog.getString();
String mergedOutputPath = Dialog.getString();

// ── Main workflow ────────────────────────────────────────────────────────────────

// Obtain the title of the already open original image
String originalFileNameWithExt = getTitle();
// Extract the file name without the extension for further processing
String originalFileName = File.nameWithoutExtension(originalFileNameWithExt);

// Construct the file name for the density filtered image
String densityFileName = originalFileName + "_density";
String extension = ".tif";

// Open the density filtered file
open(baseOutputPath + densityFileName + extension);
selectImage(densityFileName + extension);

// Process the density filtered image
run("16 colors");

// Ensure the original image is re-selected for merging
selectImage(originalFileNameWithExt);

// Merge images into a stack
run("Images to Stack", "name=[Stack] use");

// Create a maximum intensity projection of the merged stack
run("Z Project...", "projection=[Max Intensity]");

// Save the merged image as a PNG
saveAs("PNG", mergedOutputPath + originalFileName + "_merged.png");

// Close all open images
run("Close All");
