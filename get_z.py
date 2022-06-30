# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:18:17 2022

@author: lenovo
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

def get_params(x,y):
    x = sm.add_constant(x)
    md = sm.OLS(y,x)
    mdf = md.fit()
    return mdf.params

path = r"D:\tadpool\ABIDE\code\finalwar\ABIDE_Preprocess\BrainAreaData\BrainAreaData_final_clean.csv"   
brainarea_data_clean = pd.read_csv(path,index_col = 0)
site_name = brainarea_data_clean.loc[:,"SITE_ID"].value_counts().index
site_dict = dict(zip(list(site_name),np.arange(len(site_name))))
brainarea_data_clean.loc[:,"SITE_ID"] = brainarea_data_clean.loc[:,"SITE_ID"].map(site_dict)
for i in range(brainarea_data_clean.shape[0]):
    brainarea_data_clean.iloc[i,:140] = brainarea_data_clean.iloc[i,:140] / 1000
columns = list(brainarea_data_clean.columns[:140]) + ["TIV",'Sex','AgeAtScan','ADOS_SOCIAL','ADOS_COMM','ADOS_STEREO_BEHAV',"ADOS_TOTAL",'SITE_ID']
TCdata = brainarea_data_clean.loc[brainarea_data_clean.loc[:,"DxGroup"] == 2,:]
TCdata.loc[:,"AgeAtScan_2"] = TCdata.loc[:,"AgeAtScan"] ** 2
fit_params = {}
for i in (TCdata.columns[:140]):
    fit_params[i] = get_params(TCdata.loc[:,["TIV",'Sex','AgeAtScan','AgeAtScan_2']],TCdata.loc[:,str(i)])
ASDdata = brainarea_data_clean.loc[brainarea_data_clean.loc[:,"DxGroup"] == 1,:]
ASDdata = ASDdata.loc[:,columns]
ASDdata = ASDdata.loc[ASDdata.loc[:,"ADOS_COMM"].notna(),:]
ASDdata = ASDdata.loc[ASDdata.loc[:,"ADOS_SOCIAL"].notna(),:]
ASDdata = ASDdata.loc[ASDdata.loc[:,"ADOS_STEREO_BEHAV"].notna(),:]
error_index = [51456,50184,51570,50169,29118,29119,29122,29123,29125,29126,29132,29134,29137,30194
               ,30233,30231,29496,29499,29504,29512,29513,29517,29518,29520,29525,29526]
error_union = set(error_index) & set(ASDdata.index)
ASDdata = ASDdata.drop(error_union,axis = 0)
ASDdata.loc[:,"AgeAtScan_2"] = ASDdata.loc[:,"AgeAtScan"] ** 2
diffdata = ASDdata.copy()
for i in diffdata.columns[:140]:
    diffdata.loc[:,i] -= fit_params[i].loc["const"] - np.dot(diffdata.loc[:,["TIV",'Sex','AgeAtScan','AgeAtScan_2']].values,fit_params[i].iloc[1:].values)
diffdata = diffdata.iloc[:,:140]
ados_column = ['ADOS_SOCIAL','ADOS_COMM', 'ADOS_STEREO_BEHAV']

train_x = diffdata
train_y = ASDdata.loc[:,ados_column]
u,s,v = np.linalg.svd(diffdata)
train_z = u[:,:5]







