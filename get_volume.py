# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:09:11 2022

@author: lenovo
"""



import nibabel as nib
import numpy as np
import pandas as pd

img = nib.load(r"/share/inspurStorage/home1/xiedr/scripts/abide/AAL3v1_1mm.nii")
path1 = r"/share/inspurStorage/home1/xiedr/scripts/abide/code/finalwar/ABIDE_Preprocess/ZJJ_Preprocess/path/ABIDEI_path.txt"

smri_data1 = []
smri_data1_rescale = []
Anonymized_ID1 = []
f1 = open(path1,"r")
lines = f1.readlines()
for line in lines:
    line = line.replace("\n","")
    data = nib.load(line)
    line = line.split("/")
    smri_data1.append(data)
    Anonymized_ID1.append(line[-3])

smri_data1_rescale = nib.processing.resample_from_to(img,smri_data1[0],order = 0, mode = "nearest")

all_index = np.arange(1,171)
drop_index = np.arange(95,121)
need_index = np.setdiff1d(all_index,drop_index)
null_index = np.array([35,36,81,82])
need_index = np.setdiff1d(need_index,null_index)
smri_data1 = [smri_data1[i].get_fdata() for i in range(len(smri_data1))]
smri_data1_rescale = smri_data1_rescale.get_fdata()
def get_features(smri_data,smri_data_rescale,is_voxels = False):
    if not is_voxels:
        brain_area = np.zeros(need_index.shape[0])
        for k,i in enumerate(need_index):
            array_index = np.array(np.where(smri_data_rescale == i))
            sum_array = [smri_data[array_index[0,j],array_index[1,j],array_index[2,j]] for j in range(array_index.shape[1])]
            brain_area[k] = np.sum(np.array(sum_array))
        return brain_area

brain_area1_feature = np.zeros((len(smri_data1),need_index.shape[0]))
for i in range(len(smri_data1)):
    data = smri_data1[i]
    brain_area1_feature[i,:] = get_features(data,smri_data1_rescale)
BrainAreaData1 = pd.DataFrame(brain_area1_feature,index = Anonymized_ID1,columns = need_index)
BrainAreaData1.to_csv(r"/share/inspurStorage/home1/xiedr/scripts/abide/code/finalwar/ABIDE_Preprocess/ZJJ_Preprocess/BrainAreaData/BrainAreaData1.csv")



            
            
        
    


