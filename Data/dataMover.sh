#!/bin/bash

# Define the source directory containing the images (adjust as needed)
SOURCE_DIR="Data/images"

# Define the target directories (relative to the script or provide absolute paths)
SEGMENTATION_DIR="Data/segmentation1"
SEGMENTATION_DIR2="Data/segmentation2" # Corrected variable name here
ORIGINAL_DIR="Data/original"

# Create the target directories if they don't exist
mkdir -p "$SEGMENTATION_DIR"
mkdir -p "$SEGMENTATION_DIR2" # Corrected variable name here
mkdir -p "$ORIGINAL_DIR"

# Loop through all files in the source directory
for FILE in "$SOURCE_DIR"/*; do
  if [[ "$FILE" == *"__fuse"* ]]; then
    # Move files containing '__fuse' to the segmentation1 directory
    mv "$FILE" "$SEGMENTATION_DIR"/
  elif [[ "$FILE" == *"__save"* ]]; then
    # Move files containing '__save' to the segmentation2 directory
    mv "$FILE" "$SEGMENTATION_DIR2"/
  else
    # Move all other files to the original directory
    mv "$FILE" "$ORIGINAL_DIR"/
  fi
done

rmdir "$SOURCE_DIR"

echo "Files have been organized."
