#!/bin/bash

# Prompt for the directory containing the images
echo "Please input the relative location of the original images folder: "
read image_directory

# Prompt for the output directory for processed images
echo "Please input the desired output folder for predictions: "
read output_directory

# Create output directory if it doesn't exist
mkdir -p "$output_directory"

# Temporary file for storing image paths
temp_file="$(mktemp)"

# Collect all .jpg file paths in the image directory and write to temporary file
for image_path in "$image_directory"/*.jpg; do
    echo "$image_path" >> "$temp_file"
done

# Call predict.py with the temporary file as input
python predict.py --input-file "$temp_file" --output-dir "$output_directory"

# Clean up: remove the temporary file
rm "$temp_file"