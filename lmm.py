# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:36:51 2017

@author: WB512563
"""


import numpy as np
import pandas as pd

import QuantLib as ql
import TS

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class LMM(object):
    
    
    def __init__(self,YC_, tenor_ , maturity_):
        
        self.YC = YC_
        self.a = None
        self.b = None
        self.c = None
        self.d = None
    
        self.tenor = tenor_
        self.maturity = maturity_
        
        self.NumForwards = int(self.maturity / self.tenor) - 1 
        self.GPC_Param_volmatrix = np.zeros((self.NumForwards, self.NumForwards))
        
        self.correlmatrix = np.zeros((self.NumForwards,self.NumForwards))
        
        
        self.calibrated_ = False
    
    ############# set caplet_vol ##########################
    
    
    def set_caplet_vols(self, vols):
        
        self.caplet_vols = vols
        
    
    
    
    
    #############  Volatility Specification ###############    
    
    def calibrate_GPC_maturity_dependent(self):
        """
        Table 3 Brigo pg 223
        
        """
        for i in range(self.GPC_Param_volmatrix.shape[0]):
            
            j = 0
            while ( j < i +1 ):
                print "i: {}, j: {}".format(i,j)
                self.GPC_Param_volmatrix[i,j] = caplet_vols[i]
                j = j + 1
            
            print " "
            
        self.calibrated_ = True
    
    def calibrate_GPC_time_maturity_dependent(self):
        
        for i in range(self.GPC_Param_volmatrix.shape[0]):
            
            j = 0
            while ( j < i + 1):
                
                if(i == 0):
                    self.GPC_Param_volmatrix[i,j] = caplet_vols[i]
                else:
                    print "something"           #Fix this part
        
        
        pass
    
    
    def ABCD(self,t,T_n):
         
        return (self.a + self.b*(T_n - t))* np.exp( -self.c * (T_n - t) ) + self.d
    

    def ABCD_I(self, t , T_n , T_m , a , b , c , d ):
        """
        Follow West 2010   -   Interest Rate Derivatives: Lecture notes - FinanzaOnline
        """
        e_1 = np.exp( c*( t - T_n ) )
        e_2 = np.exp( c*( t - T_m ) )
        
        Comp_1 = a*d*( e_1 + e_2 ) / c
        Comp_2 = d*d*t
        Comp_3 = b*d*(  e_1*(c*(t-T_n) - 1) + e_2*(c*(t-T_m) - 1)  ) / (c*c)
        Comp_4_L = 2*a*a*c*c + 2*a*b*c*( 1 + c*(T_n + T_m -2*t) ) 
        Comp_4_R = b*b*( 1.0 + (2.0*c*c*(t-T_n)*(t-T_m)) + c*(T_n + T_m - 2.0*t) ) 
        Comp_4 = np.exp( c*(2*t - T_n - T_m) ) * (Comp_4_L + Comp_4_R) / (4.0*c*c*c)
    
        return ( Comp_1 + Comp_2 - Comp_3 + Comp_4 )

    
    def ABCD_spot_vol(self, t , T_n , T_m , a , b , c , d ):
        
        temp = self.ABCD_I(T_n , T_n , T_n , a , b , c , d) - self.ABCD_I(t, T_n , T_n , a , b , c , d)
        
        return np.sqrt( temp / T_n )
    
    
    
    ############ Correlation ###################
    
    def LMM_Correlation_Classical_two_param(self, beta):
        """
        make sure beta is >= 0      \n
        rho - p(1,M)                \n
        beta - beta                 \n
        """
        rho = 0.4
        for i in range(self.correlmatrix.shape[0]):
            for j in range(self.correlmatrix.shape[1]):
                self.correlmatrix[i][j] = rho + (1-rho)*np.exp(-beta*abs(i-j))
        
        return self.correlmatrix

    def LMM_Correlation_Rebonato_angles(self):
        pass


    ########### Swaption Approximation ##############
    
    def swaption_approx_Rebonato(self):
        
        
        
        pass

    def swaption_approx_HullWhite(self):
        
        
        
        pass

    
    
    ###########  PLotting ############################
    
    def plot_vol_term_structure(self):
        
        if self.calibrated_ == False:
            print "Not calibrated Yet"
        else:
            temp = pd.DataFrame(self.GPC_Param_volmatrix)
            temp = temp.replace(0.0, np.nan)
            
            temp.plot(title = "LMM Volatility Term Structure")
            
            
            
    def plot_correlation_surf(self):
        
        temp = pd.DataFrame(self.correlmatrix)
        
        x = temp.columns
        y = temp.index
        X,Y = np.meshgrid(x,y)
        Z = temp
        
        fig = plt.figure()
        #ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        ax.set_xlabel("T_k Forward Rate")
        ax.set_ylabel("T_k Forward Rate")        
        ax.set_zlabel("Correlation")

        ax.set_title("LMM Correlation Parametrisation")
        ax.plot_surface(X, Y, Z , cmap=cm.coolwarm,)
        
    


if __name__ == "__main__":
    
    input_location = "D:\\python\\Input\\"
    
    cols = ["Type","Tenor","Volatility"]
    data = pd.read_csv(input_location + "black76_caplet_vols" + ".csv",header = None, names = cols)
    data.Tenor =  data.Tenor.astype(float)
    #######################
    ###   Read in YC
    #######################
    filename = "ois_rates"
    cols = ["Date","Rate"]
    df1 = pd.read_csv(input_location + "YieldCurve\\" + filename + ".csv",header =None, names = cols)
    df1.Date = df1.Date.astype(int)
    df1.Date = [ql.Date(d) for d in df1.Date]
    today = ql.Date(30,ql.December,2016);
    YC = TS.yieldcurve2(today,df1)
    
    
    
    maturity = 8
    caplet_tenor = 0.5
    caplet_vols = [0.29811635, 0.359574728 , 0.395001812 , 0.412905365 , 0.419445793 , 0.418847452 , 0.413935415 , 0.406563656 , 0.397926635 , 0.38877808 , 0.379581375 , 0.370611239 , 0.36202151 , 0.353889753 , 0.346246273 ] 
    

    lmm = LMM(YC , caplet_tenor , maturity)
    lmm.set_caplet_vols(caplet_vols)
    
    

    """
    In calibration of vol:
        first need to boostrap cap vols -> caplet vols
        then calibrate caplet vols to specification
    """

    
    lmm.calibrate_GPC_maturity_dependent()
    lmm.plot_vol_term_structure()


    
    beta = 0.5
    correlation_mat = lmm.LMM_Correlation_Classical_two_param(beta)
    lmm.plot_correlation_surf()



