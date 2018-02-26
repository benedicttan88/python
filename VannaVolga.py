# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:46:18 2018

@author: Benedict Tan
"""

import numpy as np
import pandas as pd
import math



class VannaVolga(object):
    
    
    def __init__(self,YC):
        # for a time slice T

        
        self.a  = 0
        self.initialise()
    
    
    
    def initialise(self):
        pass
        
        
        

    def getVega(self, t , T , r , sigma , S , K):
                
        d1 = ( np.log(S/K) + (r + 0.5*sigma*sigma*(T-t)) )/ (sigma * np.sqrt(T-t))
        
        sigh_d1 = ( np.exp(-0.5*d1*d1) ) / np.sqrt(2.0*math.pi)
        
        return S * sigh_d1 * np.sqrt(T-t)



        
    
    
    
if __name__ == "__main__":
    
    
    
    #read the file first
    
    location = "D:\\python\\Input\\VannaVolga\\"
    
    df_vol = pd.read_excel(location + "JPYUSD.xlsx", sheetname = "Vol")
    df_YC = pd.read_excel(location + "JPYUSD.xlsx", sheetname = "YieldCurve")
    
    
    VV = VannaVolga(df_YC , df_vol)
    
    
    vol_10D_Call = []                 #collect each for each TENOR
    vol_10D_Put = []                 #collect each for each TENOR
    
    vol_ATM = []              
    vol_25D_Call = []
    vol_25D_Put = []
              
    for index, row in df_vol.iterrows():
        
        v_25_call = row["25 D BF"] + row["ATM"] + 0.5*row["25 D RR"]
        v_25_put = row["25 D BF"] + row["ATM"] - 0.5*row["25 D RR"]
        
        v_atm = row["ATM"]
        v_10_call = row["10 D BF"] + row["ATM"] + 0.5*row["10 D RR"]
        v_10_put = row["10 D BF"] + row["ATM"] - 0.5*row["10 D RR"]
        
        vol_10D_Call.append(v_10_call);  vol_10D_Put.append(v_10_put)
        vol_25D_Call.append(v_25_call);  vol_25D_Put.append(v_25_put)
        vol_ATM.append(v_atm)
        
        
    
    A = zip(vol_10D_Put, vol_25D_Call , vol_ATM , vol_25D_Call , vol_10D_Call)
        
        
        ### GO FROM TENOR TO TENOR starting from earliest maturity first
        
        
    
    
    
    
    




    
    


        
        
    