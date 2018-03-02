# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:36:00 2017

@author: tanbened
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class GBM(object):
    
    def __init__(self, S_0):
        
        self.S_0 = S_0;
        
        self.drift = 0.01;
        self.sigma = 0.1;
        
        self.NumDaysinYear = 250
        
    
    def calibrate(self):
        # Calibrate to skew --- calibrating vol    
        pass
    
    
    def mean(self, t):
        
        return self.S_0*np.exp(self.drift * t)
    
    def variance(self, t):
        
        return self.S_0*self.S_0 * np.exp(2.0*self.drift*t) * (np.exp(self.sigma*self.sigma*t) - 1.0)
        
    
    def generate_paths_analytic(self , NumSimulations , Expiry , S_0 , antihetic = True):
        
        
        pass
    
    
    def generate_paths(self , NumSimulations , Expiry , antihetic = True):
        
        
        N_Time = Expiry * self.NumDaysinYear
        
        deltaT = 1.0 / self.NumDaysinYear
        deltaT_sqrt = np.sqrt(deltaT)
        
        pathContainer = np.zeros((NumSimulations, N_Time + 1))
        pathContainer[:,0] = self.S_0                                #Fill up the first column with initial value of S_0;
        
    
        Updating_S = np.copy(pathContainer[:,0])                #this will chagne and pump into the pathContainer
        
        rng = np.random.RandomState(52)

        if antihetic == True: 
            
            for i in range(1, N_Time + 1):
                
                for j in range(NumSimulations // 2):
                    
                    rand = rng.randn()
                                        
                    #print "j: {} , i: {}".format(j,i)
                    
                    index = 2 * j
                    #print "container[{},{}]".format(index,i)
                    Updating_S[index] = Updating_S[index] + (self.drift*Updating_S[index])*deltaT + self.sigma*Updating_S[index]*rand*deltaT_sqrt
                    pathContainer[index,i] = Updating_S[index]          
                    
                    
                    index = 2*j + 1
                    #print "container[{},{}]".format(index,i)
                    Updating_S[index] = Updating_S[index] + (self.drift*Updating_S[index])*deltaT - self.sigma*Updating_S[index]*rand*deltaT_sqrt
                    pathContainer[index,i] = Updating_S[index]

                #print ""
            
            print("Path Generation Done")
            return pathContainer
            
        else:
            for i in range(1, N_Time + 1):    
                for j in range(self.NumSimulations):
                    
                    rand = rng.randn()
                    
                    Updating_S[j] = Updating_S[j] + (self.drift*Updating_S[j])*deltaT + self.sigma*Updating_S[index]*rand*deltaT_sqrt
                    pathContainer[j,i] = Updating_S[j]   
                    
            return pathContainer
                            
        
                            
    
        
        
    
    
if __name__ == "__main__":
    
    S_0 = 25
    gbm = GBM(S_0)
    
    
    NumSimulations = 1000
    Expiry = 10
    
    paths = gbm.generate_paths(NumSimulations,Expiry)
    
    ##### Printing the Lognormal Distribution ###########
    
    pd.DataFrame(paths[:,-1]).hist()
    plt.title("Distribution of S_t at T = {}".format(Expiry))
    
    ########## Plotting Mean and Variance ###############
    plt.figure()
    GBM_pathgeneration_mean = pd.DataFrame(paths).mean().tolist()
    
    N_time = Expiry * gbm.NumDaysinYear
    time = np.linspace(0,Expiry, N_time + 1)
    
    GBM_Mean = [gbm.mean(x) for x in time]
    GBM_Variance = [gbm.variance(x) for x in time]
    
    plt.title("Path of GBM Process (Mean/Variance)")
    plt.xlabel("Time (Years)"); plt.ylabel("S_t")
    plt.plot(time, GBM_Mean , label= "Theoretical Mean")
    #plt.plot(time, GBM_Variance)
    plt.plot(time, GBM_pathgeneration_mean, label = "Sample Mean {} Paths".format(NumSimulations))
    plt.legend(loc='upper left')
    
    
    
    
    #####################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    