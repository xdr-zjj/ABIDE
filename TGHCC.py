# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:46:12 2022

@author: lenovo
"""
from multiprocessing import Pool, cpu_count
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
import time
import os
import pandas as pd


ncpu = cpu_count()
def cal_correlation(x,y):
    correlation = []
    for i in range(x.shape[1]):
        correlation.append(pearsonr(x[:,i],y[:,i]))
    return correlation
def cal_pvalue(initial_correlation,permute_correlation):
    permute_correlation = np.array([[permute_correlation[i][j][0] for j in range(3)] for i in range(len(permute_correlation))])
    initial_correlation = np.array([initial_correlation[i][0] for i in range(3)])
    pvalue = (permute_correlation > initial_correlation).mean(axis = 0)
    return pvalue
def split_half(x,y,permute = 1000,nums = 10,train_ratio = 0.5):
    cca = CCA(min(x.shape[1],y.shape[1]))
    train_index = np.random.choice(x.shape[0],int(train_ratio * x.shape[0]),replace = False)
    test_index = np.delete(np.arange(x.shape[0]),train_index)#list(set(np.arange(x.shape[0])) - set(train_index))
    x_train = x[train_index,:]
    x_test = x[test_index,:]
    y_train = y[train_index,:]
    y_test = y[test_index,:]
    cca.fit(x_train,y_train)
    x_test_cca,y_test_cca = cca.transform(x_test,y_test)
    initial_correlation = cal_correlation(x_test_cca,y_test_cca)
    permute_correlation = []
    for j in range(permute):
        y_test_cca_copy = y_test_cca.copy()
        np.random.shuffle(y_test_cca_copy)
        cur_correlation = cal_correlation(x_test_cca,y_test_cca_copy)
        permute_correlation.append(cur_correlation)
        pvalue = cal_pvalue(initial_correlation,permute_correlation)
    return pvalue
def long_time_task(i):
    print('子进程: {} - 任务{}'.format(os.getpid(), i))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))

def train_test_cca_pvalue(x,y,permute = 1000,nums = 10,train_ratio = 0.5):
    #start = time.time()
    pvalues = []
    #pool = Pool(ncpu) 
    param = [[x,y,permute,nums,train_ratio] for _ in range(nums)]
    with Pool(ncpu) as pool: 
        pvalues = pool.map(split_half,param)
    """
    for i in range(nums):
        #pool.apply_async(func=split_half,args=(x,y,permute,nums,train_ratio))
        
        #pvalue = split_half(x,y,permute = 1000,nums = 10,train_ratio = 0.5)
        pvalues.append(pool.apply_async(func=split_half,args=(x,y,permute,nums,train_ratio)))
        #pool.apply_async(func = long_time_task,args = (i,))
        #print(i)
        #long_time_task(i)
    #pool.close()
    #pool.join()
    pvalues = [pvalue.get() for pvalue in pvalues]
    #end = time.time()
    #print("总共用时{}秒".format((end - start)))
    """
    return pvalues
print(train_test_cca_pvalue(np.random.normal(size = (100,5)),np.random.normal(size = (100,8))))
def sig_num_pvalue(pvalues):
    pvalues = np.array(pvalues)
    return (pvalues <= 0.05).sum(axis = 0)


class build_tree():
    def __init__(self,x,y,z,mtry,n_split,nodedepth,nodesize,permute = 1000,nums = 10,train_ratio = 0.5):
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
    def data_split(self,train_x,train_y,train_z,index,split_point):
        left_index = np.where(train_z[:,index] < split_point)[0]
        right_index = np.where(train_z[:,index] >= split_point)[0]
        left = [train_x[left_index,:],train_y[left_index,:],train_z[left_index,:]]
        right = [train_x[right_index,:],train_y[right_index,:],train_z[right_index,:]]
        return left,right
    def split_criterion_permute(self,train_x,train_y):
        pvalues = train_test_cca_pvalue(train_x,train_y,permute = self.permute,nums = self.nums,train_ratio = self.train_ratio)
        sig_p = sig_num_pvalue(pvalues)
        return sig_p

    def get_best_split(self,train_x,train_y,train_z):
        num_split_features = round(self.mtry*train_z.shape[1])
        select_features = np.random.choice(train_z.shape[1],num_split_features,replace = False)
        diff_cca = 0
        for index in select_features:
            all_splits = [np.percentile(train_z[:,index],k) for k in np.linspace(0,100,self.n_split)]
            for split_point in all_splits:
                left,right = self.data_split(train_x,train_y,train_z,index,split_point)
                if left[0].shape[0] < self.nodesize or right[0].shape[0] < self.nodesize:
                    continue
                left_cca = self.split_criterion_permute(left[0],left[1])
                right_cca = self.split_criterion_permute(right[0],right[1])
                cur_diff = max(np.abs(left_cca).max(),np.abs(right_cca).max()) * np.sqrt(left[0].shape[0] * right[0].shape[0])
                if cur_diff > diff_cca:
                     b_index, b_value, diff_cca, b_left, b_right = index, split_point, cur_diff, left, right  
        if diff_cca == 0:
            return {'index': None, 'split_point':None, 'left': [pd.DataFrame()], 'right': [pd.DataFrame()]}  
        return {'index': b_index, 'split_point':b_value, 'left': b_left, 'right': b_right}  
    def sub_split(self,root, depth):
        left = root['left']  
        right = root['right']  
        del (root['left'])  
        del (root['right'])  
        if depth == self.nodedepth:  
            root['left'] = left
            root['right'] = right
            return None
        root['left'] = self.get_best_split(left[0],left[1],left[2])  
        root['right'] = self.get_best_split(right[0],right[1],right[2])  
        if root['left']['left'][0].shape[0] == 0:# or root['left']['right'][0].shape[0] < nodesize:
            root['left'] = left
        else:
            self.sub_split(root['left'],depth + 1)  
        if root['right']['left'][0].shape[0] == 0:# or root['right']['right'][0].shape[0] < nodesize:
            root['right'] = right
        else:
            self.sub_split(root['right'],depth + 1)  
                
    def train(self):
        root = self.get_best_split(self.x,self.y,self.z)  
        self.sub_split(root,1)  
        return root         




