# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:27:43 2021

@author: eunho
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
import numpy as np

path = 'D:/한림대학교/연구실/'
test_path = path+'test_data//'

test_name = []
test_file_name = []
test_file_list = os.listdir(test_path)

for file in test_file_list:
    if file.count(".") ==1:
        name = file.split('.')[0]
        test_name.append(name[-3:])
        if name.startswith('wfdb') : test_file_name.append(name) 
       


train_data = pd.read_csv(path+'wfdb_filteredData.csv')
train_data.rename(columns={'Unnamed: 0':'name'},inplace=True)
X_train = train_data.drop(['y_train','name'],axis=1)
y_train = train_data['y_train']

import scipy.stats as ss
X_t = X_train.transpose()
z_X_train_t = ss.zscore(X_t)
z_X_train = z_X_train_t.transpose()



import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydot

dTree = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
dTree.fit(X_train, y_train)

print(dTree.score(X_train, y_train))


export_graphviz(dTree, out_file = 'dicisionTree1.dot')
#(graph, ) = pydot.graph_from_dot_file('dicisionTree1.dot', encoding = 'utf8')
#graph.write_png('dicisionTree.png')



errer = []
for i in range(0, len(test_name)-1):
    errer_cnt = 0
    test_data = pd.read_csv(test_path + test_file_name[i] + '.csv')
    test_data.rename(columns={'Unnamed: 0':'name'},inplace=True)
    X_test = test_data.drop(['y_train','name'],axis=1)
    y_true = test_data['y_train']
    
    y_pre = dTree.predict(X_test)
    
    errer.append((y_pre != y_true).sum())
    
    
    
    
   