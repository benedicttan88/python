# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:48:49 2017

@author: WB512563
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize;



def svi(Strike,Spot_F,a,b,rho,m,sigma):
    
    """
    increasing a increases general level of variance. vertical transloation of smile
    increasing b incsreases slope of put and call wings. tightening the smile
    increasing rho decreases(increases) the slope of the left(right) wing. counter clockwise rotation of smile
    increasing m increases the smile to the right
    increasing sigma reduces ATM curvature of smile
    """
    
    temp = rho * ( np.log(Strike/Spot_F) - m ) + np.sqrt( ( np.log(Strike/Spot_F) - m )**2  + sigma**2 )
    svi_sigma = a + b * temp
    
    
    return svi_sigma;


def to_minimise(market_vols):
    
    
    #Make an initial Guess from option-implied vols
    initial_a = 0.0456
    initial_b = 0.0263
    initial_rho = -0.99
    initial_sigma = 0.005
    initial_m = 0
    
    
    
    
    #suppose that market_volsp(row) = strike
    #suppose tha market_vols(col) = T
    def f(initial_a,initial_b,initial_rho,initial_m,initial_sigma):
        sum = 0;
        for i in range(market_vols.shape[0]):
            sum += ( svi(K,initial_a,initial_b,initial_rho,initial_m,initial_sigma) - market_vols[i]**2 )**2
    return sum;

    #scipy.optimize.fmin_bfgs(f,)
    
    return 0;


if __name__ == "__main__":
    
    
    vols = [60.58	,52.23,46.38,42.24,40.71,40.45,40.75,41.28,41.83,42.42,43,43.54,44.54,45.41,46.17,	46.85,47.45,47.98]
    strikes = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5.0,5.5,6.0,7,8,9,10,11,12]
    

    ###################################
    ###SVI Parameterisation Plotting
    ###################################
    
    a = 0.0410
    b = 8.9331
    m = 0.3586
    rho = 0.003060
    sigma = 0.4153
    
    a = 0.1
    b = 10.0
    #spot_F = 0.5 * ( b - a)
    spot_F = 7
    
    
    strike_list = np.linspace(a,b,100);
    paramters_list = "a = {}, b ={} , m = {}, rho ={}, sigma= {}".format(a,b,m,rho,sigma)
    plt.xlabel("Strike")
    plt.ylabel("Implied Vol")
    plt.plot(strike_list, [svi(strike,spot_F,a,b,rho,m,sigma) for strike in strike_list], label=paramters_list)
    plt.title("SVI Parametrisation  F = {}".format(spot_F));
    
    plt.figure()
    plt.plot(strikes,vols,'bo')
    
    
    
    
    
    
    
    
    
    
    
    