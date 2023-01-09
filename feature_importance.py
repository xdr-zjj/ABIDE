# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:33:38 2022

@author: lenovo
"""
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd


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


mode9_grid_search = GridSearchCV(pipe_lr,lr_param_grid,cv = skf)
mode9_perm = PermutationImportance(mode9_grid_search, n_iter=1000, cv = 5,random_state = 42).fit(mode9_train.iloc[:,:-1].values, mode9_train.iloc[:,-1].values)
mode9_sorted_id =  np.array(sorted(range(144), key=lambda k: mode9_perm.feature_importances_[k], reverse=True))
print(eli5.explain_weights(mode9_perm))

mode_concat1_grid_search = GridSearchCV(pipe_lr,lr_param_grid,cv = skf)
mode_concat1_perm = PermutationImportance(mode_concat1_grid_search, n_iter=1000, cv = 5,random_state = 42).fit(mode_concat1_train.iloc[:,:-1].values, mode_concat1_train.iloc[:,-1].values)
mode_concat1_sorted_id =  np.array(sorted(range(144), key=lambda k: mode_concat1_perm.feature_importances_[k], reverse=True))

print(eli5.explain_weights(mode_concat1_perm))

mode_concat2_grid_search = GridSearchCV(pipe_svc,svc_param_grid,cv = skf)
mode_concat2_perm = PermutationImportance(mode_concat2_grid_search, n_iter=1000, cv = 5,random_state = 42).fit(mode_concat2_train.iloc[:,:-1].values, mode_concat2_train.iloc[:,-1].values)
mode_concat2_sorted_id =  np.array(sorted(range(144), key=lambda k: mode_concat2_perm.feature_importances_[k], reverse=True))
print(eli5.explain_weights(mode_concat2_perm))

mode_concat3_grid_search = GridSearchCV(pipe_rf,rf_param_grid,cv = skf)
mode_concat3_perm = PermutationImportance(mode_concat3_grid_search, n_iter=1000, cv = 5,random_state = 42).fit(mode_concat3_train.iloc[:,:-1].values, mode_concat3_train.iloc[:,-1].values)
mode_concat3_sorted_id =  np.array(sorted(range(144), key=lambda k: mode_concat3_perm.feature_importances_[k], reverse=True))
print(eli5.explain_weights(mode_concat3_perm))

all_feature_importances = np.array([mode9_perm.feature_importances_,mode_concat1_perm.feature_importances_,mode_concat2_perm.feature_importances_,mode_concat3_perm.feature_importances_])
score_four_subtypes = pd.DataFrame(all_feature_importances.T).corr(method='pearson')
all_sorted_id = np.array([mode9_sorted_id,mode_concat1_sorted_id,mode_concat2_sorted_id ,mode_concat3_sorted_id])

sort_four_subtypes = pd.DataFrame(all_sorted_id.T).corr(method='pearson')
