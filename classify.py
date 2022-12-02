# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:22:42 2022

@author: lenovo
"""
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso

def match(mode_match,TCdata_group_compare_data):
    """
    mode_match:DataFrame,age and sex of the subtype
    TCdata_group_compare_data:DataFrame,age and sex of the TC  
    """
    TCdata_match = pd.DataFrame(np.zeros((mode_match.shape[0],3)),columns = ["Sex","AgeAtScan","index"])
    for i,index in enumerate(mode_match.index):
        cur_TCdata_group_compare_data = TCdata_group_compare_data.loc[TCdata_group_compare_data.loc[:,"Sex"] == mode_match.loc[index,"Sex"],:]
        cur_diff = cur_TCdata_group_compare_data.loc[:,"AgeAtScan"] - mode_match.loc[index,"AgeAtScan"]
        min_index = cur_diff.index[np.argmin(np.absolute(cur_diff.values))]
        TCdata_match.iloc[i,-1] = int(min_index)
        TCdata_match.iloc[i,0] = mode_match.loc[index,"Sex"]
        TCdata_match.iloc[i,1] = TCdata_group_compare_data.loc[min_index,"AgeAtScan"]
        TCdata_group_compare_data = TCdata_group_compare_data.drop(min_index,axis = 0)
    TCdata_match.index = TCdata_match.loc[:,"index"]
    TCdata_match.drop(["index"],axis = 1,inplace = True)
    return TCdata_match
def get_train(mode_data,mode_match_data,rotations,is_norm = False):
    """
    mode_data:volumetric deviations of subtype
    mode_match_data:volumetric deviations of matched TC
    """
    rotations = np.array(rotations)
    if is_norm:
        mode_mean = mode_data.mean(axis = 0)
        mode_std = mode_data.std(axis = 0,ddof = 1)
        mode_data = (mode_data - mode_mean)/mode_std
        mode_match_data = (mode_match_data - mode_mean)/mode_std
    mode_train = np.dot(mode_data.values,rotations.T)
    mode_train = np.hstack((mode_train,train_x.loc[mode_data.index,:].values))
    mode_match_train = np.dot(mode_match_data.values,rotations.T)
    mode_match_train = np.hstack((mode_match_train,TCdata_resid.loc[mode_match_data.index,:].values))
    train = np.vstack((mode_train,mode_match_train))
    train = pd.DataFrame(train,index = list(mode_data.index) + list(mode_match_data.index),columns = ["mode9","mode_concat1","mode_concat2","mode_concat3"] + list(cluster_data.columns[:140]))
    train.loc[:,"label"] = 0
    train.iloc[mode_data.shape[0]:,-1] = 1
    return train
mode9_TCdata_match = match(mode9_group_compare_data,TCdata_group_compare_data)
mode9_train = get_train(all_data[8][0],TCdata_resid.loc[mode9_TCdata_match.index,:],rotations,is_norm = False)
mode9_train.index = mode9_train.index.astype(np.int)
#mode9_X_train, mode9_X_test, mode9_y_train, mode9_y_test = train_test_split(mode9_train.iloc[:,:-1].values, mode9_train.iloc[:,-1].values, test_size=0.3, random_state=42,shuffle = True)
    

mode_concat1_TCdata_match = match(mode_concat1_group_compare_data,TCdata_group_compare_data)
mode_concat1_train = get_train(mode_concat1_x,TCdata_resid.loc[mode_concat1_TCdata_match.index,:],rotations,is_norm = False)
mode_concat1_train.index = mode_concat1_train.index.astype(np.int)
#mode_concat1_X_train, mode_concat1_X_test, mode_concat1_y_train, mode_concat1_y_test = train_test_split(mode_concat1_train.iloc[:,:-1].values, mode_concat1_train.iloc[:,-1].values, test_size=0.3, random_state=42,shuffle = True)

mode_concat2_TCdata_match = match(mode_concat2_group_compare_data,TCdata_group_compare_data)
mode_concat2_train = get_train(mode_concat2_x,TCdata_resid.loc[mode_concat2_TCdata_match.index,:],rotations,is_norm = False)
mode_concat2_train.index = mode_concat2_train.index.astype(np.int)
#mode_concat2_X_train, mode_concat2_X_test, mode_concat2_y_train, mode_concat2_y_test = train_test_split(mode_concat2_train.iloc[:,:-1].values, mode_concat2_train.iloc[:,-1].values, test_size=0.3, random_state=42,shuffle = True)

mode_concat3_TCdata_match = match(mode_concat3_group_compare_data,TCdata_group_compare_data)
mode_concat3_train = get_train(mode_concat3_x,TCdata_resid.loc[mode_concat3_TCdata_match.index,:],rotations,is_norm = False)
mode_concat3_train.index = mode_concat3_train.index.astype(np.int)
#mode_concat3_X_train, mode_concat3_X_test, mode_concat3_y_train, mode_concat3_y_test = train_test_split(mode_concat3_train.iloc[:,:-1].values, mode_concat3_train.iloc[:,-1].values, test_size=0.4, random_state=100,shuffle = True)

skf = StratifiedKFold(n_splits=5)
lr = LogisticRegression(C = 0.1)
svc = SVC(C = 100,probability = True)
rf = RandomForestClassifier(n_estimators = 10)

lr_param_grid = {"lr__C":[0.001,0.01,0.1,1,10,100,1000]}
svc_param_grid = {"svc__C":[0.01,0.1,1,10,100],"svc__gamma":[0.01,0.1,1,10,100],"svc__probability":[True]}
rf_param_grid = {"rf__n_estimators":[10,20,30]}

pipe_lr = Pipeline([('scaler', StandardScaler()), ('lr', lr)])
pipe_svc = Pipeline([('scaler', StandardScaler()), ('svc', svc)])
pipe_rf = Pipeline([('scaler', StandardScaler()), ('rf', rf)])

def get_predict_score(model,model_param_grid,x,y,nums = 100):
    """
    model:predicted model
    model_param_grid:param to search for the model
    """
    train_accuracy_score = []
    train_auc_score = []

    test_accuracy_score = []
    test_auc_score = []
    for i in range(nums):
        grid_search = GridSearchCV(model,model_param_grid,cv = skf)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=i,shuffle = True)
        grid_search.fit(X_train,y_train)
        test_accuracy_score.append(grid_search.score(X_test,y_test))
        fpr, tpr, thresholds = roc_curve(y_test, grid_search.predict_proba(X_test)[:,-1])
        test_auc_score.append(auc(fpr, tpr))
        
        train_accuracy_score.append(grid_search.score(X_train,y_train))
        fpr, tpr, thresholds = roc_curve(y_train, grid_search.predict_proba(X_train)[:,-1])
        train_auc_score.append(auc(fpr, tpr))
    return np.array(train_accuracy_score),np.array(train_auc_score),np.array(test_accuracy_score),np.array(test_auc_score),grid_search

mode9_lr_train_accuracy,mode9_lr_train_auc,mode9_lr_test_accuracy,mode9_lr_test_auc,mode9_lr_model = get_predict_score(pipe_lr,lr_param_grid,mode9_train.iloc[:,:-1].values, mode9_train.iloc[:,-1].values)
mode_concat1_lr_train_accuracy,mode_concat1_lr_train_auc,mode_concat1_lr_test_accuracy,mode_concat1_lr_test_auc,mode_concat1_lr_model = get_predict_score(pipe_lr,lr_param_grid,mode_concat1_train.iloc[:,:-1].values, mode_concat1_train.iloc[:,-1].values)
mode_concat2_svc_train_accuracy,mode_concat2_svc_train_auc,mode_concat2_svc_test_accuracy,mode_concat2_svc_test_auc,mode_concat2_svc_model = get_predict_score(pipe_svc,svc_param_grid,mode_concat2_train.iloc[:,:-1].values, mode_concat2_train.iloc[:,-1].values)
mode_concat3_rf_train_accuracy,mode_concat3_rf_train_auc,mode_concat3_rf_test_accuracy,mode_concat3_rf_test_auc,mode_concat3_rf_model = get_predict_score(pipe_rf,rf_param_grid,mode_concat3_train.iloc[:,:-1].values, mode_concat3_train.iloc[:,-1].values)
