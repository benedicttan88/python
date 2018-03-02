# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 18:02:16 2017

@author: WB512563
"""
import Settings
import pandas as pd
import numpy as np

from scipy.optimize import minimize
import scipy.optimize

import Optimization.DE as DE


class testclass(object):
    
    def __init__(self):
        self.a = 0.05               #dummy 
        self.sigma = 0.03           #dummy
        
    
    
    def first_wrap(self,w):
        
        return w*w - 2
    
    def second_wrap(self,o1):
        
        return o1*self.first_wrap(self.a)
    
    
    def third_wrap(self,o2):
        
        return o2*self.second_wrap(2)
    
    def opt(self):
        
        
        def objective_function(x):
                
            self.a = x                     
                                  
            return self.third_wrap(2)
        
        #cons = ({'type': 'ineq', 'fun': lambda x: 2*x - 1 })
        #bnds = ((0,None),(0,None))
        
        #result = minimize(objective_function,[self.a],options={'xtol':1e-8, 'disp':True} )
        #result = self.bisection(objective_function,-2,4,1e-8)
        result = DE.differential_evolution(objective_function,1,10,0,10)
        
        print result 
        
        return result 
    
    def bisection(self,f,a,b,tol):
    
        while (b-a)/2.0 >= tol:
            
            midpoint = (a+b)/2.0
            
            if f(midpoint) == 0:
                return midpoint
            
            elif f(a)*f(midpoint) < 0:
                b = midpoint
            else:
                a = midpoint
                
        return midpoint



def f(x):
	return 4*(x*x - 2)
	
def bisection(f,a,b,tol):
    
    while (b-a)/2.0 >= tol:
        
        midpoint = (a+b)/2.0
        
        if f(midpoint) == 0:
            return midpoint
        
        elif f(a)*f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
            
    return midpoint








    
if __name__ == "__main__":
    


    x = testclass()
    x.opt()
    
    print scipy.optimize.minimize(f,10)
    
    print bisection(f,-2,4,1e-8)
    
    
    