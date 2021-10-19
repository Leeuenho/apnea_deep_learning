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

model = RandomForestClassifier(oob_score=True)
model.fit(X_train,y_train)
print("data_oob_score : ",model.oob_score_)

z_model = RandomForestClassifier(oob_score=True)
z_model.fit(z_X_train,y_train)
print("std_data_oob_score : ",z_model.oob_score_)

idx_index = ['test_accuracy','Sensitivity','Precision','z_accuracy','z_Sensitivity','z_Precision']
result_data = pd.DataFrame(index = idx_index)

for i in range(0,(len(test_name)-1)):
    test_data = pd.read_csv(test_path + test_file_name[i] + '.csv')
    test_data.rename(columns={'Unnamed: 0':'name'},inplace=True)
    X_test = test_data.drop(['y_train','name'],axis=1)
    y_true = test_data['y_train']
    
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_true, pred)
    Sensitivity = recall_score(y_true,pred)
    Precision = precision_score(y_true,pred)
    
    X_test_t = X_test.transpose()
    z_X_test_t = (X_test_t - np.mean(X_test_t,axis=0)/np.std(X_test_t,axis=0))
    z_X_test = z_X_test_t.transpose()
    
    z_pred = model.predict(z_X_test)
    z_accuracy = accuracy_score(y_true,z_pred)
    z_Sensitivity = recall_score(y_true,z_pred)
    z_Precision = precision_score(y_true,z_pred)
    
    result_data.insert(result_data.shape[1], test_name[i], [accuracy,Sensitivity,Precision,z_accuracy,z_Sensitivity,z_Precision])
result = result_data.transpose()
