# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:44:01 2022

@author: lenovo
"""
import pandas as pd
import TGHCC

path = r"D:\tadpool\ABIDE\code\finalwar\ABIDE_Preprocess\YOLO\code_abide\data"
train_x = pd.read_csv(path + "\train_x.csv",index_col = 0)
train_y = pd.read_csv(path + "\train_y.csv",index_col = 0)
train_z = pd.read_csv(path + "\train_z.csv",index_col = 0)
def analysis_tree(tree):
    """
    tree:dict,result from TGHCC
    data_tree:list,store each leaf node from left to right in a tree
    """
    data_tree = []
    def get_tree(tree):
        if type(tree["left"]) == list:
            data_tree.append(tree["left"])
        else:
            get_tree(tree["left"])
        if type(tree["right"]) == list:
            data_tree.append(tree["right"])
        else:
            get_tree(tree["right"])
    get_tree(tree)
    return data_tree

def get_index(element,data_list):
    data_list = [list(data_list.values[i,:]) for i in range(data_list.shape[0])]
    element = list(element)
    return train_x.index[data_list.index(element)]

def get_membership(data_tree):
    """
    data_tree:list,result from analysis_tree
    membership:dataframe,each subject in train_x belong to which leaf node
    """
    membership = pd.DataFrame([0] * train_x.shape[0],index = train_x.index,columns = ["node"])
    for i in range(len(data_tree)):
        for j in range(data_tree[i][0].shape[0]):
            index = get_index(data_tree[i][0][j,:],train_x)
            membership.loc[index] = (i + 1)
    return membership
bt = TGHCC.build_tree()
tree = bt(train_x.values,train_y.values,train_z)        
data_tree = analysis_tree(tree)    
membership = get_membership(data_tree)

