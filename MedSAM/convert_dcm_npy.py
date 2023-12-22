# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:09:03 2023

@author: arneg
"""

import os
import numpy as np
import pydicom

def dicom_to_numpy(dicom_path):
    # Read DICOM file
    ds = pydicom.dcmread(dicom_path)

    # Extract pixel array
    pixel_array = ds.pixel_array

    return pixel_array

def save_as_npy(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all DICOM files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.dcm'):
            dicom_path = os.path.join(input_folder, filename)

            # Convert DICOM to NumPy array
            pixel_array = dicom_to_numpy(dicom_path)

            # Save NumPy array as .npy file
            output_filename = os.path.splitext(filename)[0] + '.npy'
            output_path = os.path.join(output_folder, output_filename)
            np.save(output_path, pixel_array)

if __name__ == "__main__":
    # Specify input and output folders
    input_folder = '/home/negreiro@ad.wisc.edu/project/MedSAM/data/DICOMS'
    output_folder = '/home/negreiro@ad.wisc.edu/project/MedSAM/data/numpy'

    # Run the conversion
    save_as_npy(input_folder, output_folder)
