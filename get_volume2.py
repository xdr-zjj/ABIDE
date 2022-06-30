# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:14:59 2022

@author: lenovo
"""


import nibabel as nib
import numpy as np
import pandas as pd

img = nib.load(r"/share/inspurStorage/home1/xiedr/scripts/abide/AAL3v1_1mm.nii")
path2 = r"/share/inspurStorage/home1/xiedr/scripts/abide/code/finalwar/ABIDE_Preprocess/ZJJ_Preprocess/path/ABIDEII_path.txt"

smri_data2 = []
smri_data2_rescale = []
Anonymized_ID2 = []
f2 = open(path2,"r")
lines = f2.readlines()
for line in lines:
    line = line.replace("\n","")
    data = nib.load(line)
    line = line.split("/")
    smri_data2.append(data)
    Anonymized_ID2.append(line[-3])

smri_data2_rescale = nib.processing.resample_from_to(img,smri_data2[0],order = 0, mode = "nearest")

all_index = np.arange(1,171)
drop_index = np.arange(95,121)
need_index = np.setdiff1d(all_index,drop_index)
null_index = np.array([35,36,81,82])
need_index = np.setdiff1d(need_index,null_index)
smri_data2 = [smri_data2[i].get_fdata() for i in range(len(smri_data2))]
smri_data2_rescale = smri_data2_rescale.get_fdata()
def get_features(smri_data,smri_data_rescale,is_voxels = False):
    if not is_voxels:
        brain_area = np.zeros(need_index.shape[0])
        for k,i in enumerate(need_index):
            array_index = np.array(np.where(smri_data_rescale == i))
            sum_array = [smri_data[array_index[0,j],array_index[1,j],array_index[2,j]] for j in range(array_index.shape[1])]
            brain_area[k] = np.sum(np.array(sum_array))
        return brain_area

brain_area2_feature = np.zeros((len(smri_data2),need_index.shape[0]))
for i in range(len(smri_data2)):
    data = smri_data2[i]
    brain_area2_feature[i,:] = get_features(data,smri_data2_rescale)
BrainAreaData2 = pd.DataFrame(brain_area2_feature,index = Anonymized_ID2,columns = need_index)
BrainAreaData2.to_csv(r"/share/inspurStorage/home1/xiedr/scripts/abide/code/finalwar/ABIDE_Preprocess/ZJJ_Preprocess/BrainAreaData/BrainAreaData2.csv")



            
            
        
    


