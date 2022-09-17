# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 16:53:26 2022

@author: lenovo
"""


import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
import pandas as pd


def CalCorrelation(x,y):
    correlation = []
    for i in range(x.shape[1]):
        correlation.append(pearsonr(x[:,i],y[:,i]))
    return correlation
def CalPvalue(initial_correlation,permute_correlation):
    permute_correlation = np.array([[permute_correlation[i][j][0] for j in range(3)] for i in range(len(permute_correlation))])
    initial_correlation = np.array([initial_correlation[i][0] for i in range(3)])
    pvalue = (permute_correlation > initial_correlation).mean(axis = 0)
    return pvalue
def SplitHalf(x,y,permute = 1000,nums = 10,train_ratio = 0.5):
    cca = CCA(min(x.shape[1],y.shape[1]))
    train_index = np.random.choice(x.shape[0],int(train_ratio * x.shape[0]),replace = False)
    test_index = np.delete(np.arange(x.shape[0]),train_index)#list(set(np.arange(x.shape[0])) - set(train_index))
    x_train = x[train_index,:]
    x_test = x[test_index,:]
    y_train = y[train_index,:]
    y_test = y[test_index,:]
    cca.fit(x_train,y_train)
    x_test_cca,y_test_cca = cca.transform(x_test,y_test)
    initial_correlation = CalCorrelation(x_test_cca,y_test_cca)
    permute_correlation = []
    for j in range(permute):
        y_test_cca_copy = y_test_cca.copy()
        np.random.shuffle(y_test_cca_copy)
        cur_correlation = CalCorrelation(x_test_cca,y_test_cca_copy)
        permute_correlation.append(cur_correlation)
        pvalue = CalPvalue(initial_correlation,permute_correlation)
    return pvalue

def TrainTestCcaPvalue(x,y,permute = 1000,nums = 10,train_ratio = 0.5):
    pvalues = []
    for i in range(nums):
        pvalue = SplitHalf(x,y,permute = permute,nums = nums,train_ratio = train_ratio)
        pvalues.append(pvalue)
    return pvalues
def sig_num_pvalue(pvalues):
    pvalues = np.array(pvalues)
    return (pvalues <= 0.05).sum(axis = 0)


class BuildTree():
    def __init__(self,x,y,z,mtry = 1/3,n_split = 20,nodedepth,nodesize,permute = 1000,nums = 10,train_ratio = 0.5):
        #"""
        #x:np.array,(n_samples,n_dimension1)
        #y:np.array,(n_samples,n_dimension2)
        #x,y are two sets of variables used to find multivariate correlations
        #z:np.array,(n_samples,n_dimension3),z is covariate for TGHCC division
        #mtry:float,represents what proportion of covariates to choose from z to divide each time the parent node is divided
        #n_split:int,Represents how many cutoffs are selected for each covariate selected for partitioning
        #nodedepth:int,represents the maximum depth of the tree
        #nodesize:int,Represents the maximum number of samples contained in each leaf node
        #permute:int,the number of permutations included in the permutation test
        #nums:int,Represents the number of split-half in the permutation test
        #train_ratio:float((0,1)),The proportion of the training set when cca is fitted
        #"""
        self.x = x
        self.y = y
        self.z = z
        self.mtry = mtry
        self.n_split = n_split
        self.nodedepth = nodedepth
        self.nodesize = nodesize
        self.permute = permute
        self.nums = nums
        self.train_ratio = train_ratio
    def DataSplit(self,train_x,train_y,train_z,index,split_point):
        left_index = np.where(train_z[:,index] < split_point)[0]
        right_index = np.where(train_z[:,index] >= split_point)[0]
        left = [train_x[left_index,:],train_y[left_index,:],train_z[left_index,:]]
        right = [train_x[right_index,:],train_y[right_index,:],train_z[right_index,:]]
        return left,right
    def SplitCriterionPermute(self,train_x,train_y):
        pvalues = TrainTestCcaPvalue(train_x,train_y,permute = self.permute,nums = self.nums,train_ratio = self.train_ratio)
        sig_p = sig_num_pvalue(pvalues)
        return sig_p

    def GetBestSplit(self,train_x,train_y,train_z):
        num_split_features = round(self.mtry*train_z.shape[1])
        select_features = np.random.choice(train_z.shape[1],num_split_features,replace = False)
        diff_cca = 0
        for index in select_features:
            all_splits = [np.percentile(train_z[:,index],k) for k in np.linspace(0,100,self.n_split)]
            for split_point in all_splits:
                left,right = self.DataSplit(train_x,train_y,train_z,index,split_point)
                if left[0].shape[0] < self.nodesize or right[0].shape[0] < self.nodesize:
                    continue
                left_cca = self.SplitCriterionPermute(left[0],left[1])
                right_cca = self.SplitCriterionPermute(right[0],right[1])
                cur_diff = max(np.abs(left_cca).max(),np.abs(right_cca).max()) * np.sqrt(left[0].shape[0] * right[0].shape[0])
                if cur_diff > diff_cca:
                     b_index, b_value, diff_cca, b_left, b_right = index, split_point, cur_diff, left, right  
        if diff_cca == 0:
            return {'index': None, 'split_point':None, 'left': [pd.DataFrame()], 'right': [pd.DataFrame()]}  
        return {'index': b_index, 'split_point':b_value, 'left': b_left, 'right': b_right}  
    def SubSplit(self,root, depth):
        left = root['left']  
        right = root['right']  
        del (root['left'])  
        del (root['right'])  
        if depth == self.nodedepth:  
            root['left'] = left
            root['right'] = right
            return None
        root['left'] = self.GetBestSplit(left[0],left[1],left[2])  
        root['right'] = self.GetBestSplit(right[0],right[1],right[2])  
        if root['left']['left'][0].shape[0] == 0:# or root['left']['right'][0].shape[0] < nodesize:
            root['left'] = left
        else:
            self.SubSplit(root['left'],depth + 1)  
        if root['right']['left'][0].shape[0] == 0:# or root['right']['right'][0].shape[0] < nodesize:
            root['right'] = right
        else:
            self.SubSplit(root['right'],depth + 1)  
                
    def train(self):
        root = self.GetBestSplit(self.x,self.y,self.z)  
        self.SubSplit(root,1)  
        return root         




