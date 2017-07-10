# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:18:01 2017

main.py

@author: Benedict Tan
"""
from __future__ import division
import numpy as np
import scipy
import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt

import TS
import cir_calibration

#######################
#Global Settings 
#######################

location = "D:\\worldbank\\hazard_rates_bootstrap\\input\\"
date = "31Mar2017"
today = ql.Date(31,ql.March,2017)

#read in term structure
## TS_NEW
reload(TS)
df1 = pd.read_csv(location + "ois_rates_" + date + ".csv", header=None)          #TS
cols = ["Date","Rate"]
df1.columns = cols
df1.Date = df1.Date.astype(int)
df1.Date = [ql.Date(d) for d in df1.Date]

YC = TS.yieldcurve(today,df1);


                  
                  
                  
                  
YB = TS.yieldcurve2(today,df1)

#calibrate CIR to initial term structure
reload(cir_calibration)
CIR = cir_calibration.CIR(YB);
                         

                         
                         
                         
##########
## LMM

caplet_input_location  = "D:\\python\\Input\\"
caplet_vols = pd.read_csv(caplet_input_location + "black76_caplet_vols" + ".csv", header=None)          #TS

def ABCD(t,T_n ,a,b,c,d):
    
    return (a + b*(T_n - t))* np.exp( -c * (T_n - t) ) + d


def integrate_vols(t,T_n,T_m,a,b,c,d):
    
    e_1 = np.exp( c*(t-T_n) )
    e_2 = np.exp( c*(t-T_m) )
    
    Comp_1 = a*d*( e_1 + e_2 ) /c
    Comp_2 = d*d*t
    Comp_3 = b*d*(  e_1*(c*(t-T_n) - 1) + e_2*(c*(t-T_m) - 1)  ) / c*c
    Comp_4_L = 2*a*a*c*c + 2*a*b*c*( 1 + c*(T_n + T_m -2*t) ) 
    Comp_4_R = b*b*(1 + (2*c*c*(t-T_n)*(t-T_m)) + c*(T_n + T_m - 2*t) ) 
    Comp_4 = np.exp( c*(2*t - T_n - T_m) ) * (Comp_4_L + Comp_4_R) / (4*c*c*c)
    
    
    return ( Comp_1 + Comp_2 - Comp_3 + Comp_4 )
    

def g(x):
    
    
    total_sum = 0.0;
    for i in range(caplet_vols.shape[0]):
        
        t = caplet_vols.iloc[i,1]
        #T_n = caplet_vols.iloc[i,2]
        
        #print "{} , {}".format(t,T_n)
        
        model_implied_vol = np.sqrt( ( integrate_vols(t,t,t,x[0],x[1],x[2],x[3]) - integrate_vols(0,t,t,x[0],x[1],x[2],x[3]) ) / t )
#        if integrate_vols(t,t,t,a,b,c,d) - integrate_vols(0,t,t,a,b,c,d) < 0.0:
#            print "negative sqrt"
        market_implied_vol = caplet_vols.iloc[i,2]
        total_sum += np.square( market_implied_vol - model_implied_vol )
        #total_sum = abs( market_implied_vol - model_implied_vol )
    
    return total_sum





def constr1(x):
    
    return x[0] + x[3]

def constr2(x):
    
    return x[2]

def constr3(x):
    
    return x[3]

#f(caplet_vols)




#read in caplet vols 
#optmise




if __name__ == "__main__":
    
    ##LMM
    caplet_input_location  = "D:\\python\\Input\\"
    caplet_vols = pd.read_csv(caplet_input_location + "black76_caplet_vols" + ".csv", header=None)          #TS
    
    x0 = np.array([0.01,0.02,0.01,0.02]);
    #print scipy.optimize.minimize(g,x0,options={'disp':True})
    
    
    bounds = np.array([(0,1),(0,1),(0,1),(0,1)])
    #print scipy.optimize.differential_evolution(g,bounds,disp=True)
    
    
    print "BFGS"
    print scipy.optimize.fmin_l_bfgs_b(g,x0,disp=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    