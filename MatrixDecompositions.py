# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:04:24 2017

@author: WB512563
"""

import numpy as np

#####    Matrix Decompositions


def create_correlationMatrix(obj):
    """
    Must input a list
    """
    pass
    
    
    
    
def draw_N_randomNumbers(correlation_matrix):
    
    Lower_tri = np.linalg.cholesky(correlation_matrix);
    
    size = correlation_matrix.shape[0]
    uncorrelated_var = np.random.randn(size);
    
    correlated_variables = np.dot(Lower_tri,uncorrelated_var);
                                 
    return correlated_variables;    



