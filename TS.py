# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 20:50:01 2017

@author: Benedict Tan
"""
from __future__ import division
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


    def discount(self,t, T = None):
        
        if T is None:
            r_ = self.linear_interpolate(t)
            return np.exp(-r_ * (t - self.start_date) / 365)
        else:
            print t
            print T
            r1_ = self.linear_interpolate(t)
            r2_ = self.linear_interpolate(T)
            #return np.exp( -(r2_-r1_) * (T - t) / 365 )
            return np.exp(-r2_*(T-self.start_date)/365)*np.exp(r1_*(t-self.start_date)/365)
    
        


class yieldcurve2(object):

    def __init__(self,start_date,df):
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
    
    def linear_interpolate(self,T):         #QuantLib Date
        
        if type(T) == ql.Date:
            for i in range(len(self.A)):           
                if (self.A[i][0] <= T <= self.A[i+1][0]):            
                    rate = self.A[i][1] + (T - self.A[i][0]) * (self.A[i+1][1] - self.A[i][1]) / (self.A[i+1][0] - self.A[i][0])
                    break
        else:               #not Datte, YearFractions
            for i in range(len(self.B)):
                if (self.B[i][0] <= T <= self.B[i+1][0]):            
                    rate = self.B[i][1] + (T - self.B[i][0]) * (self.B[i+1][1] - self.B[i][1]) / (self.B[i+1][0] - self.B[i][0])
                    break            
            
            #for i in range(len(A)):
                
        
        return rate


    def discount(self,t, T = None):
                
        if type(t) == ql.Date:                  #First form == Quantlib Date
            
            if T is None:
                r_ = self.linear_interpolate(t)
                return np.exp(-r_ * (t - self.start_date) / 365)
            else:
                print t
                print T
                r1_ = self.linear_interpolate(t)
                r2_ = self.linear_interpolate(T)
                #return np.exp( -(r2_-r1_) * (T - t) / 365 )
                return np.exp(-r2_*(T-self.start_date)/365)*np.exp(r1_*(t-self.start_date)/365)

        else:                                   #Second form == YearFractions
            
            if T is None:
                r_ = self.linear_interpolate(t)
                return np.exp(-r_ * t);
            else:
                r1_ = self.linear_interpolate(t)
                r2_ = self.linear_interpolate(T)
                #return np.exp( -(r2_-r1_) * (T - t) / 365 )
                return np.exp(-r2_*T)*np.exp(r1_*t)





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