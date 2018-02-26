# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:10:02 2017

@author: WB512563
"""

import pandas as pd
import numpy as np




class SVI(object):
    
    
    def __init__(self,a=0.04,b=0.02,rho=0.005,sigma=0.02,m=0.02):
        self.a = a
        self.b = b
        self.rho = rho
        self.sigma = sigma 
        #self.YC = YC
        self.m = m
        
        
    def get_TotalImpliedVariance(self,K,F):
        
        temp = (np.log(K/F) - self.m)
        inner_left = self.rho * temp
        inner_right = np.sqrt(temp*temp + self.sigma*self.sigma)
        
        return self.a + self.b * (inner_left + inner_right)         
        
        


if __name__ == "__main__":
    
    svi = SVI();
             
    strike_list = np.linspace(0.01,0.1,100);
    
    