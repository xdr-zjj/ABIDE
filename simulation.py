# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 23:15:44 2023

@author: lenovo
"""

from RFHCC import BuildTree
import numpy as np

num_simulated = 100
n_sample = 50
ans = []
for i in range(num_simulated):
    x = np.random.uniform(0,1,n_sample)
    y = np.random.uniform(0,1,n_sample)
    x_ = 1 - x
    y_ = 1 - y
    x = np.hstack((x.reshape(-1,1),x_.reshape(-1,1)))
    y = np.hstack((y.reshape(-1,1),y_.reshape(-1,1)))

    x_noise = np.random.multivariate_normal((0,0),[[1, 0], [0, 1]],size = (n_sample,))
    y_noise = np.random.multivariate_normal((0,0),[[1, 0], [0, 1]],size = (n_sample,))
    x_all = np.vstack((x,x_noise))
    y_all = np.vstack((y,y_noise))

    z = np.array(list(np.random.uniform(0,0.5,n_sample)) + list(np.random.uniform(0.5,1,n_sample)))
    z_all = z.reshape(-1,1)
    
    sim_tree = BuildTree(x_all,y_all,z_all)
    sim_root = sim_tree.train()
    ans.append(sim_root["left"][0].sum(axis = 1).sum())
