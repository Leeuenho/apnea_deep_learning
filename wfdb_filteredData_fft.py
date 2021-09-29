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

fn='a01'
name='D:/apnea-ecg-database-1.0.0 (1)/apnea-ecg-database-1.0.0//'+fn
annotation = wfdb.rdann(name,'apn')

rri_by_min = pd.DataFrame()
N=1024
xmax_lists =[] # for FFT analysis
col_lists=[]
yout=[] # A or N results
del_i=[] # temporary index to have been skipped

for i in range(annotation.ann_len):   
    start = annotation.sample[i]
    end=start + 6000    
    try: 
        qrs_index = wfdb.rdann(name,'qrs',sampfrom=start,sampto=end)        
    except:del_i.append(i); continue # skip when less than 1 min in the last data
    if qrs_index.ann_len < 35: del_i.append(i); continue # skip when less than 35 beats for cited period    
    rri_array=[]
    for ii in range(len(qrs_index.sample)-1):
        q1=1000*(qrs_index.sample[ii+1]-qrs_index.sample[ii])/annotation.fs        
        rri_array.append(q1)
    ## RRI filtering 
    den = np.median(rri_array)    
    for ii in range(1,len(rri_array)):
        cmp1=abs(rri_array[ii-1]-rri_array[ii])
        if( cmp1 > den/3): rri_array[ii]=np.mean(rri_array) # replace rri with mean rri when diff. bwt two consecutive rri values 
                                                            #    is larger than one-thirds of rri median
    for iii in range(10):
        for ii in range(len(rri_array)-1):
            cmp1=abs(rri_array[ii]-rri_array[ii+1])
            if( cmp1 > den/3): rri_array[ii]=np.mean(rri_array)
    if max(np.cumsum(rri_array)) < 40000: del_i.append(i); continue # skip when less than 40 s    
    ## end of RRI filtering                       
    x_max=max(np.cumsum(rri_array))
    x_min=min(np.cumsum(rri_array))
    xmax_lists.append(x_max) # total length of each epoch (1min) 
    ## interpolation
    flinear=interp1d(np.cumsum(rri_array),rri_array,kind='linear')
    xint=np.linspace(x_min,x_max,N,endpoint=False)
    yint=flinear(xint) ## linear interpolation output
    ywin=np.hanning(N)*yint ## applied to hanning window
    rri_by_min.insert(rri_by_min.shape[1],'xPos_'+str(int(start/100)),ywin)
    col_lists.append('xPos_'+str(int(start/100)))
    # time domain features
    # add here (mean, sdnn, rmssd, nn50, pnn50, variance) using pyhrv module
    # end of time domain
    yout.append(annotation.symbol[i]) # outut

# y_training data
y_train=[]  # y training data
for i in range(len(yout)):
    if yout[i]=='A': an=1
    else: an=0;
    y_train.append(an)
# end of y_training    

# frequency domain for HRV analysis
from scipy.fft import fft   
Xdf=pd.DataFrame()
for i in range(len(col_lists)):
    X=fft(rri_by_min[col_lists[i]].values,norm=None)
    X=np.abs(X[0:int(N/2)])    
    X=X*X/xmax_lists[i]
    X[0]=0;X[1]=X[1]/2700
    Xdf.insert(Xdf.shape[1],col_lists[i],X)
# end of frequency    

# spectral density between freq. bands
idx_lists=['lf','hf','lf/hf','lfnu','hfnu'] # add time-domain features
Xout=pd.DataFrame(index=idx_lists)
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
    Xout.insert(Xout.shape[1],col_lists[i],[lf,hf,r_value,lfnu,hfnu]) # add time-domain features
# end of spectral

# x training data after exchanging rows with columns
x_train = Xout.transpose()  
