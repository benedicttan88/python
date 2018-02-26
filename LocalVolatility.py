# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:19:34 2017

@author: Benedict Tan
"""

#######  LOCAL VOL MODEL #############################

import numpy as np
import pandas as pd

import QuantLib as ql
import TS


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# local vol need the vol surface


class LocalVolatility(object):
    

    def __init__(self):
        
        self.K = 2
        self.r = 0.05;                                              #risk free rate
    
        self.S_0 = 0.04;
        self.q = 0.0                                                #dividend
    
    
        self.VolSurface                                             #can be of SVI/SABR/Quadratic
    
        self.CapVolatiityMatrix = np.random.rand(10,10)             #Dummy variables
        self.VolStrikes  = 1                                        #Dummy variables
        self.VolMaturities = 1                                      #dummy variables
        
        
        
        
        self.calibrated_ = False
    
    
    def volSmileParametrisation():
        pass
        
    
    def diff(self):
        pass
    

    def LocalVol_Call(self):
        
        """
        Local Vol in term of Call Prices
        
        """
    
        top = differentiate(Call,T) + self.K*self.drift*differentiate(Call,K) + (rdom - self.drift)*Call
        bottom = self.K * self.K *differtiateTwice(Call,K)
    
        return top/bottom
    
    
    def LocalVol_ImpliedVol(self, K , tau):
        
        #Dupire Loval Vol at (K,tau)
        
        impl = self.VolSurface.getImpliedVol(K,tau)
        diff_tau = 0
        diff_K = 0
        diff_K_K = 0
        
        d1 =  ( np.log(self.S_0 / K) + ((self.r-self.q) + 0.5*impl*impl)* tau ) / ( impl * np.sqrt(tau) )
        
        top = impl*impl + 2.0*tau*impl*diff_tau + 2*(self.r-self.q)*K*tau*impl*diff_K
        bottom_l = np.square(1.0 + K*d1*np.sqrt(tau)*diff_K)
        bottom_r = K*K*tau*impl*( diff_K_K - d1*np.sqrt(tau)*diff_K*diff_K )
        
        return np.sqrt( top / (bottom_l + bottom_r) )
        
    
    def plot_LocalVolSurface(self):
        pass
    
    def plot_impliedVolatilitySurface(self):
        
        temp = pd.DataFrame(self.volMatrix)
        
        x = temp.columns
        y = temp.index
        X,Y = np.meshgrid(x,y)
        Z = temp
        
        fig = plt.figure()
        #ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        ax.set_xlabel("Strikes")
        ax.set_ylabel("Expiry")       
        ax.set_zlabel("Implied Vol")

        ax.set_title("Input: Implied Volatility Surface")
        ax.plot_surface(X, Y, Z , cmap=cm.coolwarm,)


    
    def generate_paths(self,NumSimulations, NumSteps):
        
        #Generate number of paths of local vol model using GBM
        
        pass
        
        

    
    
    
if __name__ == "__main__":
    
    #input_location = "C:\\Users\\Benedict Tan\\Dropbox\\Python\\Input\\"
    #input_location = "C:\\Users\\bened\\Dropbox\\Python\\Input\\"
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
    
    
    ############################ SVI Section ####################################
    
    LocalVol = LocalVolatility()
    
    """
    Igonoring the ATM Strikes here for now
    """
    ################ Input the vol surface ####################
    tempdf = pd.read_csv(input_location + "cap_vol_surface" + ".csv")
    
    lst = tempdf.columns[2:].tolist()                                                       #implied vol surface strikes
    LocalVol.volStrikes = [float(i[:-2])/100 for i in lst]
    
    lst = tempdf.Expiry.tolist()                                                            #implied vol surface expiries
    LocalVol.volExpiry = [float(i[:-2]) for i in lst]
    
    LocalVol.volMatrix = tempdf.iloc[:,2:].values
    
                                    
    
                                    
                                    
    
    
    
    
    