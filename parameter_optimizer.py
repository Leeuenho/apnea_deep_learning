# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 22:30:38 2021

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

import warnings

warnings.filterwarnings(action = 'ignore')


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


#최적의 파라미터 찾기
params = {
    'n_estimators' : [3, 11],
    'max_depth' : [i for i in range(10, 2000, 200)],
    'min_samples_leaf' : [i for i in range(10, 2000, 200)],
    'min_samples_split' : [i for i in range(10, 2000, 200)]}


from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier(oob_score=True)
grid_cv = GridSearchCV(model, param_grid = params, cv = 2, n_jobs=1, refit = True)

grid_cv.fit(X_train,y_train)

print(grid_cv.best_params_)
print(grid_cv.best_score_)


######### max_deep = 91, min_samples_leaf = 1, min_samples_split=91 
######### accuracy = 0.647




                
                
                

