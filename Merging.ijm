// Define the base paths for input, output, and merged directories
String baseInputPath = "//input_data/";
String baseOutputPath = "//output_data/density/";
String mergedOutputPath = "//merged/";

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
