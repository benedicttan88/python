# -*- coding: utf-8 -*-
"""
Created on Fri Jan 05 09:15:12 2018

@author: tanbened
"""

from __future__ import division
import pandas as pd
import numpy as np


###############
### YIELDCURVE NO QUANTLIB INVOVLED

###############

class yieldcurve(object):
    
    def __init__(self,start_date,df):                                   #start_date = int , df= [int, zerorates]
        self.start_YF = 0.0;
        self.start_date = start_date
        self.Date = df.Date
        
        
        
        self.YF = [ (d - start_date)/365 for d in df.Date]
        self.Rate = df.Rate
        self.Original_Set = zip(self.Date,self.Rate)
        
        self.A = self.Original_Set
        self.B = zip(self.YF,self.Rate)
        if self.start_date < self.Date[0]:
            self.A = [(self.start_date,self.Rate.values[0])] + self.A
            self.B = [(self.start_YF,self.Rate.values[0])] + self.B
    
    
    
    def linear_interpolate(self,T):                                     #Linear Interpolation
    
        for i in range(len(self.B)):
            if (self.B[i][0] <= T <= self.B[i+1][0]):            
                rate = self.B[i][1] + (T - self.B[i][0]) * (self.B[i+1][1] - self.B[i][1]) / (self.B[i+1][0] - self.B[i][0])
                break           
                          
        return rate                  



    def discount(self,t, T = None):
        
        if T is None:
            r_ = self.linear_interpolate(t)
            return np.exp(-r_ * t);
        else:
            r1_ = self.linear_interpolate(t)
            r2_ = self.linear_interpolate(T)
            #return np.exp( -(r2_-r1_) * (T - t) / 365 )
            return np.exp(-r2_*T)*np.exp(r1_*t)        
    
    
    def forward(self,T1,T2):
        
        #note that t <= S
        if ((T2==0) or (T1 == T2)):
            return 0
        else:
            df = self.discount(T1,T2)
        
            return ( (1/df) - 1 ) / (T2-T1)           
        
    
    def get_r(self,T1,T2):
        
        if ((T2 == 0) or (T1 == T2)):
            return 0
        else:
            df = self.discount(T1,T2)
        
            return -np.log(df)/(T2-T1)            
    
    
    def get_swap_rate(self,t,T1,T2,cap_freq):
        
        top = self.discount(t,T1) - self.discount(t,T2)
        
        bottom = 0.0
        tempT = T2
        while (tempT >= (T1 + cap_freq)):
            bottom += cap_freq * self.discount(t,tempT)
            tempT -= cap_freq

        return top/bottom       
        

    
    
    
if __name__ == "__main__":
        
    print "testing"
    
    location = "C:\\Users\\tanbened\\python\\Input\\"
                
    df = pd.read_csv(location + "YieldCurve\\ " + "ois_rates" + ".csv", header=None)
    cols = ["Date","Rate"]
    df.columns = cols
    df.Date = df.Date.astype(int)
    
    
    start_date = 42738
    
    YC = yieldcurve(start_date,df)
    


    #df.Date = [ql.Date(d) for d in df.Date]
    
    #today = ql.Date(31,ql.March,2017)
    
    #YC = yieldcurve(today,df);
    #print YC.A
                
                
                