# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 12:54:51 2017

@author: WB512563
"""

## HW1F Constant Parameters

import TS
import numpy as np
import pandas as pd

import QuantLib as ql
from scipy.stats import norm

from scipy.optimize import minimize
from scipy.optimize import differential_evolution


class HW1F_constant(object):
    
    """
    Constant Parameters
    Following Brigo's implementation in Book Pg75
    """
    
    
    def __init__(self,YC_):
        self.a = 0.01               #dummy 
        self.sigma = 0.03           #dummy
        self.YC = YC_
        self.Lognormal = True
        self.shift = 0.0;
        self.blackPrices = 0
        
        self.temp_list =0
        
        self.calibrated_ = False
    
    def shift_a(self,shift):
        self.a += shift
        
    def shift_sigma(self,shift):
        self.sigma += shift
    
    
    
    def V(self,t,T):
        temp_ = np.exp(-self.a*(T-t))
        return (T-t) + (2/self.a)*temp_  - (temp_*temp_/(2*self.a)) - (3/(2*self.a))
    
    
    def A(self,t,T):
        
        left = self.YC.discount(0,T) / self.YC.discount(0,t)
        
        inner_left = self.B(t,T) * self.YC.forward(0,t)
        inner_right =  self.sigma * self.sigma / (4*self.a) * (1 - np.exp(-2*self.a*t)) * self.B(t,T) * self.B(t,T)
        right = np.exp(inner_left - inner_right)
    
        return left * right
        

    def B(self,t,T):
        
        return (1 - np.exp(-self.a*(T-t) )) / self.a
        
   
    def r(self,t):

        return self.YC.get_r(0,1e-8)
        
        
    def P(self,t,T):
        
        return self.A(t,T) * np.exp( -self.B(t,T) * self.r(t))
    
    
    
    def __HW_Caplet(self,Notional, t, T1, T2, K):
        """
        T1 = Fixing Time
        T2 = End Time
        K = Strike
        Defines the HW Caplet in Brigos Book Pg 76
        """
        theta = T2 - T1
        strike = 1/(1+(K*theta))
        hw_caplet = Notional*(1 + K*theta)*self.__ZBP(t,T1,T2,strike)
        return hw_caplet
    
    
    def __ZBP(self,t,T1,T2,K):
        """
        T1 = T
        T2 = S
        K = X
        """
        inner = (1 - np.exp(-2*self.a*(T1-t))) / (2 * self.a)
        sigma_p = self.sigma * np.sqrt(inner) * self.B(T1,T2)
        
        h =  (( np.log( self.YC.discount(t,T2) / (self.YC.discount(t,T1)*K) ) )/sigma_p) + 0.5*sigma_p 
        
        zbp = K*self.YC.discount(t,T1)*norm.cdf(-h + sigma_p) - (self.YC.discount(t,T2)*norm.cdf(-h))
        
        return zbp
    

    

    def __Black_Caplet(self,Notional, t, T1, T2, vol, K):
        
        #Lognormal Pricing
        
        Fwd = self.YC.forward(T1,T2)
        
        d1 = ( np.log(Fwd/K) + 0.5*vol*vol*T1 ) /  ( vol * np.sqrt(T1) )
        d2 = d1 - ( vol * np.sqrt(T1) )
        
        return Notional*(T2-T1)*self.YC.discount(0,T2)*(Fwd*norm.cdf(d1) - K*norm.cdf(d2))
    
    
    
    def function_to_minimise(self,lst):
        
        sum_ = 0
        
        for (T1,T2,K,blackPrices) in lst:
            sum_ += np.square((blackPrices/self.__HW_Caplet(1,0,T1,T2,K)) - 1)
        
        
        
        return sum_
         
        

    def calibrate(self,black_vols):
        
        blackPrices =[]
        maturities_T2 = []
        strikes = []
        #convert the caplet vol to black_caplet prices
        for (x,y) in black_vols:
            T1 = x-0.5
            T2 = x
            K = self.YC.forward(T1,T2)
            maturities_T2.append(x)
            strikes.append(K)
            blackPrices.append(self.__Black_Caplet(1,0,x-0.5,x,y,K))
        
        maturities_T1 = [(x - 0.5) for x in maturities_T2]     #freq is set to 0.5
        lst = list(zip(maturities_T1,maturities_T2,strikes,blackPrices))
        #x = [a, sigma]
        self.temp_list = lst
        self.blackPrices = blackPrices
        print( lst )
        
        def objective_function(x, lst):
            
            #self.a = x[0]
            self.sigma = x
            
            return self.function_to_minimise(lst)
            
         
        
        result = minimize(objective_function,[self.sigma],args=(lst),method= "Powell",options={'xtol':1e-8, 'disp':True} )
        #bounds = [(0,5),(0,5)]
        #result = differential_evolution(objective_function,bounds,args=(lst))
                
        print( result )
        
        self.sigma = result.x
        
        self.calibrated_ = True
        
        return lst
        
    def calibration_report(self):
        
        if self.calibrated_ == False:
            return "Not Calibrated"
        else:
            modelPrices = []
            for (T1,T2,K,blackPrices) in self.temp_list:
                caplet_price = self.__HW_Caplet(1,0,T1,T2,K)
                modelPrices.append(caplet_price)
            
            calibration_report = pd.DataFrame(list(zip(self.blackPrices,modelPrices)),columns=["Market Caplet Price","HW Caplet Price"])
            
            return calibration_report
        
        
        
    


if __name__ == "__main__":
    
    
    #location = "D:\\python\\Input\\"
    #df = pd.read_csv(location + "Yieldcurve\\" + "ois_rates" + ".csv", header=None)
    df = pd.read_csv("Input/Yieldcurve/ois_rates.csv", header= None)
    cols = ["Date","Rate"]
    df.columns = cols
    df.Date = df.Date.astype(int)
    
    df.Date = [ql.Date(d) for d in df.Date]
    
    
    today = ql.Date(30,ql.December,2016)
    
    YC = TS.yieldcurve2(today,df);
    
    HW = HW1F_constant(YC)
    
    
    
    
    
    
    
    #######  Caplet vol location
    #black_caplet_vols = pd.read_csv(location + "Caplet Vols\\" + "Caplet ATM Vol from Cap Object" + ".txt",delimiter='\t')
    black_caplet_vols = pd.read_csv("Input/Caplet Vols/Caplet ATM Vol from Cap Object" + ".txt",delimiter= '\t' )
    
    mat=[]; vol =[]
    for index,row in black_caplet_vols.iterrows():
        mat.append(row[0])
        vol.append(row[1])
    
    black_vols = zip(mat,vol)
    
    
    HW.calibrate(black_vols)
    calibration_report = HW.calibration_report()
    calibration_report.plot()
    

    simulations = 100
    timesteps = 10
    
        
    
    
    
        
        
    