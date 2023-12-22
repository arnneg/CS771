# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:07:05 2023

@author: Ariana_Negreiro
"""

#Functions
def Mask_to_Array(file_path):
    img = Image.open(file_path)
    array = np.array(img)

    if array[0,0] != 0:
        indices_zero = array == 0
        indices_one = array == 1

        array[indices_one] = 0
        array[indices_zero] = 1

    return array

def Array_to_STL(array):
   
    verts,faces,normals,values = measure.marching_cubes(array)
    stl = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        stl.vectors[i] = verts[f]
    return stl

#Turn Masks output from the Automatic Segmentation Model into STLs or Part Files to inport into Mimics
from glob import glob
import os, pdb
import numpy as np
import matplotlib.pyplot as plt
from Computer_Vision_Functions import Mask_to_Array, Array_to_STL

Mask_Folder_Path = '../masks/'
STL_Output_Folder = '../STLs/'
#Mask_Image_Path = '../masks/BPH009_Phase001_0022.gif'

Subject_Name = input('Please input the subject id i.e. (BPH009): ')
Phase_Number = input('Please input the phase number i.e. (Phase001): ')

#Subject_Name = Mask_Image_Path.split('/')[2].split('.')[0].split('_')[0]
#Phase_Number = Mask_Image_Path.split('/')[2].split('.')[0].split('_')[0]


Array_3D = np.empty(shape=(256,256,140))
Mask_List =  glob(os.path.join(Mask_Folder_Path,"{:s}_{:s}*").format(Subject_Name,Phase_Number))
Starting_Mask = int(Mask_List[0].split('_')[-1].split('.')[0])
#End_Mask = int(Mask_List[-1].split('_')[-1].split('.')[0])
for i in range(len(Mask_List)):
    Array_2D = Mask_to_Array(Mask_List[i])
    Array_3D[:,:,i+Starting_Mask] = Array_2D 
stl = Array_to_STL(Array_3D)

STL_Output_Folder = STL_Output_Folder + Subject_Name
STL_Output_Exists = os.path.exists(STL_Output_Folder)

if STL_Output_Exists is False:
    os.makedirs(STL_Output_Folder)

stl.save('{:s}/{:s}_{:s}.stl'.format(STL_Output_Folder,Subject_Name,Phase_Number))