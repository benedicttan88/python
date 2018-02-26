# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:19:04 2017

@author: WB512563
"""

#Smoothing the Hazard Rates


from __future__ import division
import pandas as pd
import numpy as np
import scipy.optimize
import QuantLib as ql
import matplotlib.pyplot as plt
from collections import OrderedDict

import sklearn
from sklearn.linear_model import LinearRegression

def short(lmbda,tau):
    temp = lmbda*tau
    return (1 - np.exp(-temp))/temp

def medium(lmbda,tau):
    temp = lmbda*tau
    return ((1 - np.exp(-temp))/temp) - np.exp(-temp) 


def NS(b1,b2,b3,lmbda,tau):
    
    return b1 + b2*short(lmbda,tau) + b3*medium(lmbda,tau)

location = "D:\\python\\CVA_Proxy\\Input\\"
df = pd.read_csv(location+ "BootstrappedHazardRates.csv",index_col=0)

time_to_maturity=  [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]


################### Nelson Siegel Fitting Area ################################
lambda_ = 0.0609

spread_list = pd.DataFrame()

for name, values in df.iteritems():

    raw_hazard_rates = values.tolist()
    A = zip(time_to_maturity,raw_hazard_rates)
    for i, (x,y) in enumerate(A):
        #print x
        A[i] =  A[i] + (1,short(lambda_,x),medium(lambda_,x))
    dd = pd.DataFrame(A)
    dd.dropna(inplace = True)
    b = dd.iloc[:,1].values
    W = dd.iloc[:,2:].values
               
    reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    reg.fit(W,b)
    NS_coef = reg.coef_

    NS_Curve = pd.DataFrame([NS(NS_coef[0],NS_coef[1],NS_coef[2],lambda_,i) for i in time_to_maturity])

    spread_list = pd.concat([spread_list,NS_Curve],axis=1)
    
    
spread_list.columns = df.columns
spread_list.index = time_to_maturity





