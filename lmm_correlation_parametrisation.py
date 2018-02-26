# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:48:59 2017

@author: WB512563
"""




import numpy as np
import matplotlib.pyplot as plt



a = np.zeros(shape = (10,10))

#2 parameters
# p(1,M)
# B

def LMM_Classical_two_param(M,rho,beta):
    
    """
    M - MxM correlation matrix
    rho - p(1,M)
    beta - beta
    
    """
    
    matrix = np.zeros(shape=(M,M))
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i][j] = rho + (1-rho)*np.exp(-beta*abs(i-j))
            
    return matrix


    



if __name__ == "__main__":
       
    M = 10
    rho = 0.04
    beta = 1
    
    
    corr = LMM_Classical_two_param(M,rho,beta)
    
    


        
    



