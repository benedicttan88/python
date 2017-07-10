# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:34:01 2017

#CIR Calibration


@author: Benedict Tan
"""

import numpy as np
import pandas as pd
import scipy


#dr(t) =  a(b - r(t)) * dt + sigma sqrt(r(t)) dZ

   
class CIR(object):
    
    def __init__(self,YC_):
        
        self.YC = YC_
        self.alpha = 0.25
        self.beta = 0.05
        self.sigma = 0.01
        

    def A(self,t,T):
        
        gamma = np.sqrt(self.alpha**2 + 2*(self.sigma**2))
        top = 2 * gamma * np.exp( 0.5*(self.alpha + gamma)*(T-t) )
        bottom = (gamma + self.alpha) * ( np.exp(gamma*(T-t)) - 1 ) + 2*gamma
                 
        pow_term = 2*self.alpha*self.beta / (self.sigma**2)
        
        result = np.power(top/bottom, pow_term)
        
        return result 
    
   
    def B(self,t,T):
        
        gamma = np.sqrt(self.alpha**2 + 2*(self.sigma**2))
        exp_term = np.exp(gamma*(T-t))
        
        top = 2 * (exp_term - 1)
        bottom = (gamma + self.alpha)*(exp_term - 1.0) + 2*gamma
                 
        return top/bottom;


    def P(self,t,T):
       deltaT = 1.0/365.0;
       # r0 = self.YC.
       r0 = self.YC.discount(0,deltaT);
       
       return self.A(t,T) * np.exp(-self.B(t,T) * r0)
    
    
    
    def calibrate(self,initial_a=0.25, initial_b= 0.05, initial_sigma= 0.05):
        
        self.alpha = initial_a;
        self.beta = initial_b
        self.sigma = initial_sigma
        
        YC_df = []
        for i in range(len(self.YC.YF)):
            YC_df.append(self.YC.discount(self.YC.YF[i]));
        
        deltaT = 1/365.0;
        total_sum = 0.0;
        for i in range(len(YC_df)):
            total_sum += abs(YC_df[i] - self.P(0,deltaT))
        
        return total_sum;        



    def calibrate2(self,x):
        
        self.alpha = x[0];
        self.beta = x[1];
        self.sigma = x[2];
        
        YC_df = []
        for i in range(len(self.YC.YF)):
            YC_df.append(self.YC.discount(self.YC.YF[i]));
        
        deltaT = 1/365.0;
        total_sum = 0.0;
        for i in range(len(YC_df)):
            total_sum += abs(YC_df[i] - self.P(0,deltaT))
        
        return total_sum; 
    
    
    
    
    def cal(self,y):
        
        self.alpha = 0.3
        self.beta = 0.05
        self.sigma = 0.05
        
        x0 = [self.alpha,self.beta,self.sigma]
        print "x0: {}".format(x0);
                   
                   
        bounds = [(0,1),(0,1),(0,1)]
        #con2 = {'type': 'ineq', 'fun': self.cons}
        
        #result = scipy.optimize.fmin_bfgs(self.calibrate2,x0)
        result = scipy.optimize.fmin_slsqp(self.calibrate2,x0)
        
        
        return result
    
    def cons(self,x):
        
        self.alpha = x[0]
        self.beta = x[1]
        self.sigma = x[2]
        
        #return 2*self.alpha*self.beta - (self.sigma*self.sigma)
        return 2*x[0]*x[1] - (x[2]*x[2])
    
#        def objective_function(self,YC_df,self.alpha,self.beta,self.sigma):
#            
#            total_sum = 0.0;
#            deltaT = 1/365.0;
#            
#            for i in range(len(YC_df)):
#                total_sum += abs(YC_df[i] - self.P(0,deltaT))
#        
#            return total_sum;
#        
#        bounds = []
#        result = scipy.optimize.differential_evolution(objective_function,bounds)
#        
#        return result;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
            
            
        
        
        
        
        
        
    