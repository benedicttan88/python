# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:28:49 2017

@author: tanbened
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import TS

import GBM

"""
What do you need

matrix of stock prices
discount curve
your payoff function

-paramters
    basis functions
    # of basis functions
    
"""

class LongstaffSchwartz(object):
    
    
    def __init__(self, paths, payoffF):
        
        self.paths = paths
        self.payoffF = payoffF
        #self.YC = YC_
        
        self.callability_schedule = []
    
    
    def payoff(self,S_T):
        
        K = 1.1
        return max( K - S_T , 0)
    
    
    def Laguerre(self,x, degree):
        
        if degree == 1:
            return 1.0;
        
        elif degree == 2:
            return 1.0 - x;
    
        elif degree == 3:
            return 0.5*( 2.0 - 4*x + x*x )
        
        elif degree == 4:
            return 
        
    
    
    
    
    
#    def price(self, 
#              path_container,
#              callability_schedule,
#              YC):
#        
#        main_index = []
#            
#        for i in range(self.paths.shape[1]-1,0,-1):
#            print i
#            
#            payoffs = np.array( [self.payoff(x) for x in paths[:,i]] )
#            main_index = np.where(payoffs > 0)                                              #main indices that will change later
#            
                                 
                                 
                                 
        
        
        
            
                                 
                                 
                                 
                                 
            
        pass
    
    
if __name__ == "__main__":
    
    
    ######### YC #########################
    
    
    
    ######### GBM PROCESS ################
    
    S_0 = 25
    gbm = GBM.GBM(S_0)
    
    NumSimulations = 1000
    Expiry = 10
    
    paths = gbm.generate_paths(NumSimulations,Expiry)
        
        
    ######################################
    
    def payoff(x):
        
        return 2*x + 2;
        
    ls = LongstaffSchwartz(paths,payoff)
    

            