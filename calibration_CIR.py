# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:34:01 2017

#CIR Calibration


@author: Benedict Tan
"""
from __future__ import division
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

import QuantLib as ql
import TS

from scipy.optimize import minimize
from scipy.optimize import differential_evolution

import Optimization.DE as DE

from scipy.stats import norm

#dr(t) =  a(b - r(t)) * dt + sigma sqrt(r(t)) dZ
#a = k
#b = theta
   
class CIR_constant(object):
    
    def __init__(self,YC_):
        
        self.YC = YC_
        self.alpha = 0.2
        self.beta = 0.04
        self.sigma = 0.07
        self.calibrated_ = False
        

    def shift_speed__(self,shift):
        self.alpha += shift
    
    def shift_mean__(self,shift):
        self.beta += shift

    def shift_vol__(self,shift):
        self.sigma += shift


        
    def A(self,t,T):
        """
        Defined in Brigos Book pg 66
        """
        
        gamma = np.sqrt(self.alpha*self.alpha + (2*self.sigma*self.sigma))
        top = 2 * gamma * np.exp( 0.5*(self.alpha + gamma)*(T-t) )
        bottom = (gamma + self.alpha) * ( np.exp(gamma*(T-t)) - 1 ) + 2*gamma
                 
        pow_term = 2*self.alpha*self.beta / (self.sigma*self.sigma)
        
        tol = 1e-8
        if (pow_term < tol):
            print "pow: {}  \t SMALL".format(pow_term)
            
        
        temp = top/bottom
        result = np.power(temp, pow_term)
        
        return result 
    
   
    def B(self,t,T):
        """
        Defined in Brigos Book pg 66
        """
        gamma = np.sqrt(self.alpha*self.alpha + (2*self.sigma*self.sigma))
        exp_term = np.exp(gamma*(T-t))
        
        top = 2 * (exp_term - 1)
        bottom = (gamma + self.alpha)*(exp_term - 1.0) + 2*gamma
                 
        return top/bottom;

    def r(self,t):

        return self.YC.get_r(0,1/365.0)

    def P(self,t,T):
        """
        Defined in Brigos Book pg 66
        """
       
        return self.A(t,T) * np.exp(-self.B(t,T) * self.r(t))
        

    def function_to_minimise(self):
        """
        Minimising Discount Factors
        Fitting to Term Structure
        """
        YC_df = []
        for i in range(len(self.YC.YF)):
            YC_df.append(self.YC.discount(self.YC.YF[i]));
                        
        sum_ = 0.0
        for i in range(len(YC_df)):
            sum_ += abs( (YC_df[i] - self.P(0,i)) / YC_df[i] )
            #sum_ += np.square( (YC_df[i]/self.P(0,i)) - 1 )
        
        penalty = self.penalty()
        lambda_ = 1000.0
        #return sum_
        return sum_ + lambda_*penalty
    
    def penalty(self):
        
        return np.square(max(0, (self.sigma*self.sigma - 2*self.alpha*self.beta)))
    
    def meng_calibrate(self):
        
        #np.random.RandomState.set_state(('MT19937', keys, pos))
        np.random.seed(22)
        # 0.1, 0.01, 0.01
        
        initial_a_vol = 0.1
        initial_b_vol = 0.01
        initial_vol_vol = 0.01
        
        optimal_diff = 1000
        
        gamma = 2
        NumSimulation = 500
        
        optimal_a = 0.25
        optimal_b = 0.05
        optimal_vol = 0.075
        
        
        for k in range(NumSimulation):
            
            scale_factor = np.exp(-gamma*k/NumSimulation)
            a_vol = initial_a_vol * scale_factor
            b_vol = initial_b_vol * scale_factor
            vol_vol = initial_vol_vol * scale_factor
                    
            while True:
                
                self.alpha = optimal_a + norm.ppf(np.random.uniform()) * a_vol
                self.beta = optimal_b + norm.ppf(np.random.uniform()) * b_vol
                self.sigma = optimal_vol + norm.ppf(np.random.uniform()) * vol_vol
                                                   
                while (self.alpha <= 0):
                    self.alpha = optimal_a + norm.ppf(np.random.uniform()) * a_vol
                while (self.beta <= 0):
                    self.beta = optimal_b + norm.ppf(np.random.uniform()) * b_vol
                while (self.sigma <= 0):
                    self.sigma = optimal_vol + norm.ppf(np.random.uniform()) * vol_vol                
                
                condition = (2 * self.alpha * self.beta) - ( self.sigma * self.sigma )
                
                if(condition > 0):
                    break
                


            temp_diff = 0.0
            
            YC_df = []
            for i in range(len(self.YC.YF)):
                YC_df.append(self.YC.discount(self.YC.YF[i]));
                        
            for i in range(len(YC_df)):
                temp_diff += abs((YC_df[i] - self.P(0,i)))
            
            
            if (temp_diff < optimal_diff):
                print "global diff: {}".format(temp_diff)
                optimal_diff = temp_diff
                optimal_a = self.alpha
                optimal_b = self.beta
                optimal_vol = self.sigma
            
        #Local Search
        for k in range(NumSimulation):
            
            scale_factor = np.exp(-gamma*k/NumSimulation)
            a_vol = initial_a_vol / 10 * scale_factor
            b_vol = initial_b_vol / 10 * scale_factor
            vol_vol = initial_vol_vol  / 10 * scale_factor

            while True:
                
                self.alpha = optimal_a + norm.ppf(np.random.uniform()) * a_vol
                self.beta = optimal_b + norm.ppf(np.random.uniform()) * b_vol
                self.sigma = optimal_vol + norm.ppf(np.random.uniform()) * vol_vol
                                
                while (self.alpha <= 0):
                    self.alpha = optimal_a + norm.ppf(np.random.uniform()) * a_vol
                while (self.beta <= 0):
                    self.beta = optimal_b + norm.ppf(np.random.uniform()) * b_vol
                while (self.sigma <= 0):
                    self.sigma = optimal_vol + norm.ppf(np.random.uniform()) * vol_vol                
                
                condition = (2 * self.alpha * self.beta) - ( self.sigma * self.sigma )
                
                if(condition > 0):
                    break
     
            
            temp_diff = 0.0
            
            YC_df = []
            for i in range(len(self.YC.YF)):
                YC_df.append(self.YC.discount(self.YC.YF[i]));
                        
            for i in range(len(YC_df)):
                temp_diff += abs((YC_df[i] - self.P(0,i)))
            
            
            if (temp_diff < optimal_diff):
                print "temp_diff: {}".format(temp_diff)
                optimal_diff = temp_diff
                optimal_a = self.alpha
                optimal_b = self.beta
                optimal_vol = self.sigma
            
        
        #end
        self.alpha = optimal_a
        self.beta = optimal_b
        self.sigma = optimal_vol
        
        self.calibrated_ = True
        
        
        condition_satisfied =  2*self.alpha*self.beta - self.sigma*self.sigma
        if condition_satisfied > 0:
            print "condition satisfied: {}".format(condition_satisfied)
        else:
            print "condition not satisfied: {}".format(condition_satisfied)
            
            
            
        return condition_satisfied
    
            
                                        
    
    
    def calibrate(self):
        
        def objective_function(x):
            
            print "a: {} , b: {} , sigma: {}".format(x[0],x[1],x[2])
            self.alpha = x[0]
            self.beta = x[1]
            self.sigma = x[2]
            
            
            return self.function_to_minimise()
        
        initial_parameters = [self.alpha, self.beta, self.sigma]
        cons = ({'type': 'ineq', 'fun': lambda x: (2*x[0]*x[1]) - x[2]*x[2] })
        bnds = ((1e-4,1),(1e-4,1),(1e-4,1))
        minimizer_kwargs =  {"bounds":bnds}
        
        
        #result = minimize(objective_function,initial_parameters,method='SLSQP',bounds =bnds, options={'xtol':1e-8, 'disp':True} )
        result = differential_evolution(objective_function,bounds= bnds)
        #result = scipy.optimize.basinhopping(objective_function, initial_parameters)
        #result = DE.differential_evolution(objective_function,3,10,0,1)
        
        print result
        
        #set the result to the parameters
        self.alpha = result.x[0]
        self.beta = result.x[1]
        self.sigma = result.x[2]
        
        self.calibrated_ = True
        
        condition_satisfied =  2*self.alpha*self.beta - self.sigma*self.sigma
        if condition_satisfied > 0:
            print "condition satisfied: {}".format(condition_satisfied)
        else:
            print "condition not satisfied: {}".format(condition_satisfied)
        
        pass
    
    def calibration_report__(self):
        
        if self.calibrated_ == False:
            return "Not Calibrated"
        else:
            print "Model Calibrated!"
            YC_Yearfrac = YC.YF
            YC_df = []
            model_df = []
            for i in range(len(self.YC.YF)):
                T = self.YC.YF[i]
                YC_df.append(self.YC.discount(T));
                model_df.append(self.P(0,T))
                
            calibration_report = pd.DataFrame(zip(YC_df,model_df),columns=["Market DF","Model DF"],index=YC_Yearfrac)
            
            return calibration_report



if __name__ == "__main__":
    

    location = "D:\\python\\Input\\"
    
    
    df = pd.read_csv(location + "Yieldcurve\\" + "ois_rates" + ".csv", header=None)
    cols = ["Date","Rate"]
    df.columns = cols
    df.Date = df.Date.astype(int)
    
    df.Date = [ql.Date(d) for d in df.Date]
    
    
    today = ql.Date(30,ql.December,2016)
    
    YC = TS.yieldcurve2(today,df);
    
    CIR = CIR_constant(YC)

    
    CIR.calibrate()
    #CIR.meng_calibrate()

    calibration_report = CIR.calibration_report__()
    calibration_report.plot()

    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
            
            
        
        
        
        
        
        
    