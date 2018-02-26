# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:25:39 2017

@author: WB512563
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt



def f(rho,z):
    
    X_z_top = ( np.sqrt(1 - 2*rho*z + z*z) + z - rho ) 
    
    return np.log( X_z_top / (1- rho) )




if __name__ == "__main__": 
    
    z = 10
    
    left_end = -1 + 1e-8
    right_end = 1 - 1e-8
    
    t = np.linspace(left_end,right_end,1000)    
    y = [f(rho,z) for rho in t]
    
    
    plt.plot(t,y)
    
    
    
    
    