import os, pdb
import numpy as np
import scipy as sc
from PIL import Image
from glob import glob

def Horizontal_Flip(img_path,img_save_folder):
    img = Image.open(img_path)
    flipped_image = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img_name = img_path.split('/')[2].split('.')[0]
    img_file_extension = img_path.split('/')[2].split('.')[1]
    img_save_path = img_save_folder + img_name + '_Flipped.' + img_file_extension
    return flipped_image.save(img_save_path)

def Image_Rotation(img_path,rotation_degree,img_save_folder):
    img = Image.open(img_path)
    rotated_image = img.rotate(rotation_degree)
    img_name = img_path.split('/')[2].split('.')[0]
    img_file_extension = img_path.split('/')[2].split('.')[1]
    img_save_path = img_save_folder + img_name + '_Rotated.' + img_file_extension
    return rotated_image.save(img_save_path)

def Mask_Rotation(mask_path,rotation_degree,mask_save_folder):
    mask = Image.open(mask_path)
    rotated_mask = mask.rotate(rotation_degree,fillcolor=1)
    mask_name = mask_path.split('/')[2].split('.')[0]
    mask_file_extension = mask_path.split('/')[2].split('.')[1]
    mask_save_path = mask_save_folder + mask_name + '_Rotated.' + mask_file_extension
    return rotated_mask.save(mask_save_path)

def Hausdorff_Distance(original_mask_path,predicted_mask_path):
    Original = Image.open(original_mask_path)
    Predicted = Image.open(predicted_mask_path)
    Original = np.array(Original)
    Predicted = np.array(Predicted)
    Distance,orig_idx,predicted_idx = sc.spatial.distance.directed_hausdorff(Original,Predicted)
    return Distance, orig_idx, predicted_idx

def DICE_Coeff(original_mask_path,predicted_mask_path):
    Original = Image.open(original_mask_path)
    Predicted = Image.open(predicted_mask_path)
    Original = np.array(Original)
    Predicted = np.array(Predicted)
    Intersection = np.sum(Original * Predicted)
    DICE = (2. * Intersection) / (np.sum(Original) + np.sum(Predicted))
    return DICE
