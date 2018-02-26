# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 20:50:01 2017

@author: Benedict Tan
"""

import pandas as pd
import numpy as np
import QuantLib as ql

#location = "D:\\worldbank\\hazard_rates_bootstrap\\input\\"

#df = pd.read_csv(location + "ois_rates_31Mar" + ".csv", header=None)
#cols = ["Date","Rate"]
#df.columns = cols
#df.Date = df.Date.astype(int)
#
#df.Date = [ql.Date(d) for d in df.Date]

def discount(T,df,start_date):
    
    A = zip(df.Date,df.Rate)
    
    
    if start_date < df.Date[0]:
        A = [(start_date,df.Rate.values[0])] + A
    
    for i in range(len(A)):        
        if (A[i][0] < T < A[i+1][0]):            
            rate = A[i][1] + (T - A[i][0]) * (A[i+1][1] - A[i][1]) / (A[i+1][0] - A[i][0])


    #np.exp(-rate * )    
    
    return rate   
    
class yieldcurve(object):

    def __init__(self,start_date,df):
        self.start_date = start_date
        self.Date = df.Date
        self.Rate = df.Rate
        self.A = zip(self.Date,self.Rate)
    
    def linear_interpolate(self,T):         #QuantLib Date
        
        A = zip(self.Date,self.Rate)
        if self.start_date < self.Date[0]:
            A = [(self.start_date,self.Rate.values[0])] + A
        
        for i in range(len(A)):        
            
            if (A[i][0] <= T < A[i+1][0]):            
                rate = A[i][1] + (T - A[i][0]) * (A[i+1][1] - A[i][1]) / (A[i+1][0] - A[i][0])
                break
                
        return rate

    def discount(self,T):
        r_ = self.linear_interpolate(T)
        return np.exp(-r_ * (T - self.start_date) / 365)
        
    

        
        
#if __name__ == "__main__":
        
    
#    df = pd.read_csv(location + "ois_rates_31Mar" + ".csv", header=None)
#    cols = ["Date","Rate"]
#    df.columns = cols
#    df.Date = df.Date.astype(int)
#    
#    df.Date = [ql.Date(d) for d in df.Date]
#    
#    
#    today = ql.Date(31,ql.March,2017)
#    
#    YC = yieldcurve(today,df);