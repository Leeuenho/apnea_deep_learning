import numpy as np
import wfdb
from scipy.interpolate import interp1d
import pandas as pd
import pyhrv
import os

fn_list = []
file_name = []
path = 'C:\\Users\\daehun\\Desktop\\파이썬\\HRV프로그램\\apnea-ecg-database-1.0.0\\apnea-ecg-database-1.0.0\\'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.apn')]
Total_Xout = pd.DataFrame()
del_i=[] # temporary index to have been skipped
for i in range(len(file_list_py)):
    if len(file_list_py[i])==7:
        fn_list.append(file_list_py[i])
        
for file in fn_list:
    if file.count(".") ==1:
        name = file.split('.')[0]
        file_name.append(name)

for k in range(35,70):
    
    name=path+file_name[k]
    annotation = wfdb.rdann(name,'apn')
    
    rri_by_min = pd.DataFrame()
    N=1024
    xmax_lists =[] # for FFT analysis
    col_lists=[]
    yout=[] # A or N results
    mean=[]
    i_mean=[]
    sdnn = []
    rmssd = []
    i_nn50 = []
    nn50 = []
    pnn50 = []
    variance = []
    i_variance = []
    
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
            if q1 < 350: del_i.append(i); continue
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
        if max(np.cumsum(rri_array)) < 40000 or max(np.cumsum(rri_array)) > 62000: del_i.append(i); continue # skip when less than 40 s    
        ## end of RRI filtering                       
        x_max=max(np.cumsum(rri_array))
        x_min=min(np.cumsum(rri_array))
        xmax_lists.append(x_max) # total length of each epoch (1min) 
        
        ## interpolation
        flinear=interp1d(np.cumsum(rri_array),rri_array,kind='linear')
        xint=np.linspace(x_min,x_max,N,endpoint=False)
        yint=flinear(xint) ## linear interpolation output
        ywin=np.hanning(N)*yint ## applied to hanning window
        rri_by_min.insert(rri_by_min.shape[1],str(file_name[k])+'xPos_'+str(int(start/100)),ywin)
        col_lists.append(str(file_name[k])+'xPos_'+str(int(start/100)))
        # time domain features
        rri_mean = pyhrv.time_domain.nni_parameters(rri_array)
        i_mean += rri_mean
        rri_sdnn = pyhrv.time_domain.sdnn(rri_array)
        sdnn += rri_sdnn
        rri_rmssd = pyhrv.time_domain.rmssd(rri_array)
        rmssd += rri_rmssd
        rri_nn50 = pyhrv.time_domain.nn50(rri_array)
        i_nn50 += rri_nn50
        rri_variance = pyhrv.time_domain.nni_differences_parameters(rri_array)
        i_variance += rri_variance
        # add here (mean, sdnn, rmssd, nn50, pnn50, variance) using pyhrv module
        # end of time domain
        yout.append(annotation.symbol[i]) # outut
    for j in range(1,len(i_mean),4):
        mean.append(i_mean[j])
    
    for jj in range(0,len(i_nn50)):
        if jj%2==0:
            nn50.append(i_nn50[jj])
        else:
            pnn50.append(i_nn50[jj])
    for jjj in range(0,len(i_variance),3):
        variance.append(i_variance[jjj])
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
    idx_lists=['lf','hf','lf/hf','lfnu','hfnu','mean','sdnn','rmssd','nn50','pnn50','variance','y_train'] # add time-domain features
    
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
        Xout.insert(Xout.shape[1],col_lists[i],[lf,hf,r_value,lfnu,hfnu,mean[i],sdnn[i],rmssd[i],nn50[i],pnn50[i],variance[i],y_train[i]]) # add time-domain features
    Total_Xout = pd.concat([Total_Xout,Xout],axis=1)

    Xout.reset_index(drop=True)
    # end of spectral
    # x training data after exchanging rows with columns
    x_train = Xout.transpose()
    x_train.to_csv('wfdb_tset_filteredData_fft_result_'+file_name[k]+'.csv')