# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:44:01 2022

@author: lenovo
"""
import pandas as pd
def analysis_tree(tree):
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
    """
    element:一维array
    data_list:dataframe,shape:(571,140)
    """
    data_list = [list(data_list.values[i,:]) for i in range(data_list.shape[0])]
    element = list(element)
    return train_x.index[data_list.index(element)]

def get_membership(data_tree):
    membership = pd.DataFrame([0] * train_x.index,index = train_x.index,columns = ["node"])
    for i in range(len(data_tree)):
        for j in range(data_tree[i][0].shape[0]):
            index = get_index(data_tree[i][0][j,:],train_x)
            membership.loc[index] = (i + 1)
    return membership

data_tree = analysis_tree(tree)    
membership = get_membership(data_tree)

