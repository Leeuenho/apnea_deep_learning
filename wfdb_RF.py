# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:38:21 2021

@author: eunho
"""

import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings(action = 'ignore')

#data 가저오기
fl = "D:/한림대학교/연구실/git_lab//"
fn = "wfdb_filteredData.csv"
data = pd.read_csv(fl+fn)

#입력 X와 결과 Y 분리
labels = np.array(data['y_train'])
features = data.drop('y_train', axis = 1)

#data 이름 제거
features = data.drop('Unnamed: 0', axis = 1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.4, shuffle = True)

#모델 생성
model = RandomForestClassifier(oob_score = True)

#모델학습
model.fit(x_train, y_train)

#모델 검증
predicted = model.predict(x_test)

#훈련세트 정확도 
train_acc = model.score(x_train, y_train)

#테스트 정확도
test_acc = model.score(x_test, y_test)

#OOB 샘플 정확도
oob_acc = model.oob_score


print("훈련세트 정확도 : {:.3f}".format( train_acc))
print("테스트세트 정확도 : {:.3f}".format( test_acc))
print("OOB 샘플 정확도 : {:.3f}".format( oob_acc))


#confusion matrix 구현

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

k = confusion_matrix(y_test, predicted, labels)

label = ['apnea', 'non-apnea']
plot = plot_confusion_matrix(model, x_test, y_test,
                             display_labels = label,
                             normalize = None)
