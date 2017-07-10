# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:36:51 2017

@author: WB512563
"""


import numpy as np



class LMM(object):
    
    
    def __init__(self,YC_):
        
        self.YC = YC_
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        
    
    def ABCD(self,t,T_n):
         
        return (self.a + self.b*(T_n - t))* np.exp( -self.c * (T_n - t) ) + self.d
    
    
    
    def integrate_vols(t,T_n,T_m,a,b,c,d):
    
        e_1 = np.exp( c*(t-T_n) )
        e_2 = np.exp( c*(t-T_m) )
        
        Comp_1 = a*d*( e_1 + e_2 ) /c
        Comp_2 = d*d*t
        Comp_3 = b*d*(  e_1*(c*(t-T_n) - 1) + e_2*(c*(t-T_m) - 1)  ) / c*c
        Comp_4_L = 2*a*a*c*c + 2*a*b*c*( 1 + c*(T_n + T_m -2*t) ) 
        Comp_4_R = b*b*(1 + (2*c*c*(t-T_n)*(t-T_m)) + c*(T_n + T_m - 2*t) ) 
        Comp_4 = np.exp( c*(2*t - T_n - T_m) ) * (Comp_4_L + Comp_4_R) / (4*c*c*c)
    
    
        return ( Comp_1 + Comp_2 - Comp_3 + Comp_4 )
    
    
    
    def ATM_vol(self):
        """
        a
        b
        c
        d
        """
        
        a + b() * np.exp( -c * (T_n - t))
        
        
    
        
        
        
    
    vol_param()    
    
    
    
    
    
    