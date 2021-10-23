# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:27:43 2021

@author: eunho
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sc


import warnings
warnings.filterwarnings('ignore')

path = 'D:/한림대학교/연구실/'

test_path = path+'test_data//'

test_name = []
test_file_name = []
test_file_list = os.listdir(test_path)

for file in test_file_list:
    if file.count(".") ==1:
        name = file.split('.')[0]
        test_name.append(name[-3:])
        test_file_name.append(name)

train_data = pd.read_csv(path+'wfdb_filteredData.csv')
train_data.rename(columns={'Unnamed: 0':'name'},inplace=True)
X_train = train_data.drop(['y_train','name'],axis=1)
y_train = train_data['y_train']

X_t = X_train.transpose()
X_t = sc.zscore(X_t)
X_t = X_t.transpose()

model = RandomForestClassifier(oob_score=True)
model.fit(X_t,y_train)
print("data_oob_score : ",model.oob_score_)

Xdata_out=[]
for i in range(0, (len(test_name)-1)):
    test_data = pd.read_csv(test_path+test_file_name[i]+'.csv')
    test_data.rename(columns={'Unnamed: 0':'name'},inplace=True)
    X_test = test_data.drop(['y_train','name'],axis=1)
    
    X_test = X_test.transpose()
    X_test = sc.zscore(X_test)
    X_test = X_test.transpose()
    pred = model.predict(X_test)
    ap=0;non=0
    for k in range(len(pred)):
        if pred[k]==1: ap +=1;
        if pred[k]==0: non +=1;
    Xdata_out.append(np.round(ap/(non+1)*100,1))

Xdata_y=np.array([63,37.7,0.13,0,34,0,21,48,18.5,10,5,33,18.7,79.5,
                  15.9,24,0,0,56.2,43,19,0,14.3,0,48,15.1,75,75,0,41,93.5,71.8,0.13,0.38,0])
Xdata_yt=np.array([1,1,0,0,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,1,1,1,1,0,1,1,1,0,0,0])

xt=np.arange(1,36,1)    
plt.rcParams['figure.figsize']=(10,5)
plt.plot(xt,Xdata_out,'o',color='b')
plt.plot(xt,Xdata_yt*50,'o',color='r')
#plt.plot(xt,Xdata_y,'x')
plt.axhline(y=30)
plt.xticks(xt)
