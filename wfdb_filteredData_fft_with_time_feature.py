# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 15:35:32 2021

@author: user
"""

# from IPython.display import display
# import matplotlib.pyplot as plt
# import os
# import shutil
# import posixpath
# from wfdb import processing

import numpy as np
import wfdb
from scipy.interpolate import interp1d
import pandas as pd
import biosppy
import pyhrv.tools as tools
import pyhrv.time_domain as td
from pyhrv.hrv import hrv
import pyhrv



fn='a01'
name='D:/apnea-ecg-database-1.0.0 (1)/apnea-ecg-database-1.0.0//'+fn
annotation = wfdb.rdann(name,'apn')

rri_by_min = pd.DataFrame()
N=1024
xmax_lists =[] # for FFT analysis
col_lists=[]
yout=[] # A or N results
del_i=[] # temporary index to have been skipped

# mean, sdnn, rmssd, nn50, pnn50

mean = []
sdnn = []
rmssd=[]
nn50 = []
pnn50 = []
variance = []


for i in range(annotation.ann_len):   
    start = annotation.sample[i]
    end=start + 6000   
    '''
    너무 짧거나 문제가 있어서 data 추출이 불가하여 에러가 날 때 원래는 프로그램 중지
    근데 10번째 에러, 11번 째 정상일 수 있음
    그래서 에러가 날 경우 종료하는게 아니라 except하고 이어서 실행
    finally 옵션도 존재 -> 에러가 발생함에도 불구하고 실행해야할 것 정의
    '''
    try:  
        qrs_index = wfdb.rdann(name,'qrs',sampfrom=start,sampto=end)
    except:   # skip when less than 1 min in the last data
        del_i.append(i); 
        continue 
    if qrs_index.ann_len < 35: del_i.append(i); continue # skip when less than 35 beats for cited period   
    '''
    왜 35개미만을 제한?? 
    심박수가 몹시 낮은 사람 -> 데이터 분석이 어려움
    혹은 노이즈 등의 문제로인해 qrs제대로 못찾아낼거 대비 제거
    '''
    rri_array=[]
    for ii in range(len(qrs_index.sample)-1): ##RR interval 구하기
        q1=1000*(qrs_index.sample[ii+1]-qrs_index.sample[ii])/annotation.fs   #point to seconds
        ''' 1000ms = 1s'''
        rri_array.append(q1)
        
    ## RRI filtering  
    ''' RR interval 사이 차 구하는 이유 / 중앙값이랑 비교하는 이유 -> 노이즈 제거...?
    ***노이즈 제거
    사람마다 심박수 차이 존재하기때문에 사람따라 달라지도록 median을 기준으로 구분
    '''
    den = np.median(rri_array) #중앙 값 구하기
    
    for ii in range(1,len(rri_array)):
        cmp1=abs(rri_array[ii-1]-rri_array[ii])
        if( cmp1 > den/3): rri_array[ii]=np.mean(rri_array) # replace rri with mean rri when diff. bwt two consecutive rri values 
                                                            #    is larger than one-thirds of rri median
    
    '''왜 10번만 ?? 위의 코드와 차이
    
        비정상 data많을 경우 위 한번 코드로 부족할 수 있기때문에 반복
    '''
    for iii in range(10):
        for ii in range(len(rri_array)-1):
            cmp1=abs(rri_array[ii]-rri_array[ii+1])
            if( cmp1 > den/3): rri_array[ii]=np.mean(rri_array)
            
            ''' 40000 이유 그리고 왜 40000보다 작으면 skip? 
                vlf까지 구하려면 최소 40초 이상의 데이터가 필요하기때문
            '''
    if max(np.cumsum(rri_array)) < 40000: del_i.append(i); continue # skip when less than 40s     && np.cumsum = 누적합계
    ## end of RRI filtering        

               
    x_max=max(np.cumsum(rri_array))
    x_min=min(np.cumsum(rri_array))
    xmax_lists.append(x_max) # total length of each epoch (1min) 
    
    
    ## interpolation
    flinear=interp1d(np.cumsum(rri_array),rri_array,kind='linear')
    xint=np.linspace(x_min,x_max,N,endpoint=False) #N = 1024
    yint=flinear(xint) ## linear interpolation output
    
    ## applied to hanning window
    ywin=np.hanning(N)*yint 
    
    rri_by_min.insert(rri_by_min.shape[1],'xPos_'+str(int(start/100)),ywin) #dataFrame.insert(data 추가할 위치, 추가할 열의 이름, 추가할 data)
    col_lists.append('xPos_'+str(int(start/100)))
    # time domain features
    # add here (mean, sdnn, rmssd, nn50, pnn50, variance) using pyhrv module
    # end of time domain
    
    '''
    sdnn = 표준편차
    rmssd = 제곱 차의 평균 제곱근
    nn50  = 50ms 이상 차이나는 rr interval (숫자 변경 -> 그 data 찾아냄)
    pnn50 = nn50 / 전체 
    variance
    '''
    
    mean.append(td.nni_parameters(rri_array)['nni_mean'])
    sdnn.append(td.sdnn(rri_array)['sdnn'])
    rmssd.append(td.rmssd(rri_array)['rmssd'])
    nn50.append(td.nn50(rri_array)['nn50'])
    pnn50.append(td.nn50(rri_array)['pnn50'])
    variance.append(td.nni_differences_parameters(rri_array)['nni_diff_mean'])

    
    yout.append(annotation.symbol[i]) # outut 정상인 데이터만 annotation갖고와야하기때문에 제일 늦게 해야함

# y_training data
y_train=[]  # y training data
for i in range(len(yout)): #yout = annotation symbol
    # 무호흡 = 1, 정상 = 0
    if yout[i]=='A': an=1
    else: an=0;
    y_train.append(an)
# end of y_training    

# frequency domain for HRV analysis
from scipy.fft import fft   
Xdf=pd.DataFrame()
for i in range(len(col_lists)):
    X=fft(rri_by_min[col_lists[i]].values,norm=None) ##ywin
    X=np.abs(X[0:int(N/2)])    
    X=X*X/xmax_lists[i]
    X[0]=0
    X[1]=X[1]/2700          
    Xdf.insert(Xdf.shape[1],col_lists[i],X)
# end of frequency    

# spectral density between freq. bands
idx_lists=['lf','hf','lf/hf','lfnu','hfnu','mean','sdnn', 'rmssd', 'nn50', 'pnn50', 'variance'] # add time-domain features
Xout=pd.DataFrame(index=idx_lists)

#해당 주파수의 point 찾기..?
''' 1을 더하는 이유 '''
for i in range(len(col_lists)): 
    vlf_lo=int((xmax_lists[i]/N)*0.003333)+1
    vlf_hi=int((xmax_lists[i]/N)*0.04)
    lf_lo=int((xmax_lists[i]/N)*0.04)+1
    lf_hi=int((xmax_lists[i]/N)*0.15)
    hf_lo=int((xmax_lists[i]/N)*0.15)+1
    hf_hi=int((xmax_lists[i]/N)*0.4)
    vlf_sum=0;lf_sum=0;hf_sum=0    
    for ii in range(len(Xdf)):
        if(ii >= vlf_lo and ii < vlf_hi):
            vlf_sum += Xdf[col_lists[i]][ii]
        elif(ii >= lf_lo and ii < lf_hi):
            lf_sum += Xdf[col_lists[i]][ii]
        elif(ii >= hf_lo and ii < hf_hi):
            hf_sum += Xdf[col_lists[i]][ii]
        else:
            continue
    vlf=np.log(vlf_sum)
    lf=np.log(lf_sum)
    hf=np.log(hf_sum)
    tp=np.log(vlf_sum+lf_sum+hf_sum)    
    r_value=lf/hf
    lfnu=lf/(lf+hf)
    hfnu=hf/(lf+hf)
    Xout.insert(Xout.shape[1],col_lists[i],[lf,hf,r_value,lfnu,hfnu,mean[i], sdnn[i], rmssd[i], nn50[i], pnn50[i], variance[i]]) # add time-domain features
# end of spectral

# x training data after exchanging rows with columns
x_train = Xout.transpose()  
