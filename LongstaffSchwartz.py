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


class LongstaffSchwartz(object):
    
    
    def __init__(self, paths):
        
        self.paths = paths
        #self.YC = YC_
        
        self.callability_schedule = []
    
    
    def payoff(self,S_T):
        
        K = 1.1
        return max( K - S_T , 0)
    
    
    def price(self, 
              path_container,
              callability_schedule,
              YC):
        
        main_index = []
            
        for i in range(self.paths.shape[1]-1,0,-1):
            print i
            
            payoffs = np.array( [self.payoff(x) for x in paths[:,i]] )
            main_index = np.where(payoffs > 0)                                              #main indices that will change later
            
                                 
                                 
                                 
        
        
        
            
                                 
                                 
                                 
                                 
            
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

    ls = LongstaffSchwartz(paths)
    ls.price()

            