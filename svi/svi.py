# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:43:32 2017

@author: Benedict Tan
"""

import numpy as np
import pandas as pd

import QuantLib as ql
import TS


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class SVI(object):
    
    def __init__(self):
        
        ###### SVI Raw Parameters ###################
        
        self.a = 2
        self.b = 0.05;
        self.rho = -0.05;
        
        self.m = 0.05                                               #first fixing
        self.sigma = 0.5                                            #first fixing
    
        self.c = self.b*self.sigma
        self.d = self.b*self.rho*self.sigma
    
        ##### Implied Vol surface ####################
    
        self.CapVolatiityMatrix = np.random.rand(10,10)             #Dummy variables
        self.VolStrikes  = 1                                        #Dummy variables
        self.VolMaturities = 1                                      #dummy variables
        
        
        #############################################################################################
        
        self.calibrated_ = False
        
        
        
        
    def RawParametrization(self, x):
        """
        Following Zeliade SVI Paper
        K = strike
        
        """
        #x = np.log(K/S)             #log forward moneyness
                  
        return self.a + self.b*( self.rho*(x-self.m) + np.sqrt(np.square(x-self.m) + self.sigma*self.sigma) )

    def RawParametrization_Transformed(self,x):
        """
        transforming for y = (x-m) / sigma
        """
        y = (x - self.m) / self.sigma
        
        return self.a + self.b*self.sigma * (self.rho*y + np.sqrt(y*y + 1.0))
        
    
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


    def function_to_minimise_inner(self):
        
        timeslice = 1;
        marketvols = [20.06,22.1,19.98,19.97,23.77,25.6,27.22,29.94,32.15,34.01,35.59,36.99,38.23,39.34,40.34,41.27,42.12]
        S_0 = 20
        strikes = [1.0,1.5,2.0,3.0,3.5,4.0,5.0,6,7,8,9,10,11,12,13,14.0]
        strikes = [i/100 for i in strikes]
        log_moneyness = [np.log(x/S_0) for x in strikes]
        log_moneyness = [0] + log_moneyness                                                                 #adding the ATM log forward moneyness
        
        sum_ = 0.0
        for i in range(len(marketvols)):
            sum_ += np.square( self.RawParametrization_Transformed(log_moneyness[i]) - marketvols[i] )
        
        return sum_
    
    def calibrate(self):

        
        timeslice = 1;
        marketvols = [20.06,22.1,19.98,19.97,23.77,25.6,27.22,29.94,32.15,34.01,35.59,36.99,38.23,39.34,40.34,41.27,42.12]
        S_0 = 20
        strikes = [1.0,1.5,2.0,3.0,3.5,4.0,5.0,6,7,8,9,10,11,12,13,14.0]
        strikes = [i/100 for i in strikes]
        log_moneyness = [np.log(x/S_0) for x in strikes]
        log_moneyness = [0] + log_moneyness                                                                 #adding the ATM log forward moneyness
            
        print "x: {}".format(log_moneyness) 
        
        ### Inner Optimisation
        def objective_function_inner(x):
            
            self.a = x[0]
            self.b = x[1]
            self.rho = x[2]
            
            
            print "m: {}".format(self.m)
            
            return self.function_to_minimise_inner()
        
        """
        d = |rho b sigma|
        c = b sigma
        """
        
        initial_parameters = [self.a, self.b, self.rho]
        max_total_variance = timeslice*max(marketvols)
        cons = ({'type': 'ineq', 'fun': lambda x: (x[1]*self.sigma) - 4*self.sigma },                                   # c <= 4*sigma
                {'type': 'ineq', 'fun': lambda x: abs(x[2]*x[1]*self.sigma) - x[1]*self.sigma },                        #|d| <= c
                {'type': 'ineq', 'fun': lambda x: abs(x[2]*x[1]*self.sigma) - (4.0*self.sigma - x[1]*self.sigma) },     #|d| <= 4*sigma - c
                {'type': 'ineq', 'fun': lambda x: x[0] - max_total_variance } )                                         # a <= max(marketvols)
                
        bnds = ((1e-4,10),(-1000,1000),(1e-4,1))
        minimizer_kwargs =  {"bounds":bnds}

        #result = minimize(objective_function_inner,initial_parameters,method= "Powell",options={'xtol':1e-8, 'disp':True} )
        result = minimize(objective_function_inner,initial_parameters,method='SLSQP',bounds =bnds, constraints= cons,options={'disp':True} )


        ##End Inner Optimisation
        
        
        
        
        

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
    
    LocalVol = SVI()
    
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
                                    
                                    
    LocalVol.plot_impliedVolatilitySurface()                                
                                    
                                    
    ########################### CALIBRATION OF SVI ################################
    
    LocalVol.calibrate();
                                    
                                    
                                    
                                    
                                    