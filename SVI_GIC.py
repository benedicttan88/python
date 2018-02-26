# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:43:32 2017

@author: Benedict Tan
"""
import sys

import numpy as np
import pandas as pd

#import TS
import TS_noQL

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.optimize import minimize
from scipy.optimize import differential_evolution

class SVI(object):
    
    def __init__(self):
        
        ###### SVI Raw Parameters ###################
        
        self.ahat = abs(np.random.randn())                              #
        self.c = abs(np.random.randn())
        self.d = abs(np.random.randn())
    
        self.m = np.random.uniform();                                                #fixed
        self.sigma = np.random.uniform();                                           #fixed
        
        #self.m = 0.01
        #self.sigma = 0.01
                                      
        self.a = 0.0
        self.b = 0.0                                                #dummy
        self.rho = 0.0                                              #dummy
    
        ##### Implied Vol surface ####################
    
        self.CapVolatiityMatrix = np.random.rand(10,10)             #Dummy variables
        self.VolStrikes  = 1                                        #Dummy variables
        self.VolMaturities = 1                                      #dummy variables
        
        self.data = 1                                               #Dummy variable
        #############################################################################################
        
        self.calibrated_ = False
        
        
    def get_paramS(self):
        
        print "a: {} , b: {} , rho: {} , m: {} , sigma: {}".format(self.a,self.b,self.rho,self.m,self.sigma)
        
    def RawParametrization(self, x):
        """
        Following Zeliade SVI Paper
        K = strike
        gives you the total implied variance
        """
        #x = np.log(K/S)             #log forward moneyness
                  
        return self.a + self.b*( self.rho*(x-self.m) + np.sqrt(np.square(x-self.m) + self.sigma*self.sigma) )

    def RawParametrization_Transformed(self,y):
        """
        transforming for y = (x-m) / sigma
        gives you the total implied variance
        """
        # note that y = (x - self.m) / self.sigma
        
        return self.ahat + self.d*y + self.c * ( np.sqrt((y*y) + 1.0) )
        
    
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

    ##########################    
    
    def function_min_outer(self):
        
        T = self.data[0][0]                                      #timeslice at T
        marketVols = [z for (x,y,z,a,b) in self.data]            #market implied Vol
        log_moneyness = [b for (x,y,z,a,b) in self.data]         #get the list of log_moneyness
        
        assert(len(marketVols) == len(log_moneyness))
                         
        cost_outer = 0.0
        for i in range(len(marketVols)):
            cost_outer += np.square(self.RawParametrization(log_moneyness[i]) - T*marketVols[i]*marketVols[i])
        
        return cost_outer


    def satisfy_inner_constraints(self,temp_a, temp_d, temp_c , total_impl_variance):
        eps = sys.float_info.epsilon
        
        cond1 = (-eps <= temp_c) and (temp_c <= 4*self.sigma + eps)
        cond2 = (-eps <= temp_a) and (temp_a <= total_impl_variance.max() + eps)
        cond3 = abs(temp_d) <= min(temp_c , 4*self.sigma - temp_c)
        
        #print "Condition for c: {}".format(cond1)
        #print "Condition for a: {}".format(cond2)
        #print "Condition for d: {}".format(cond3)
        
        return (cond1 and cond2 and cond3)
    
    def cost_function(self, y, a, d, c, total_impl_variance):
        
        diff = a + d*y + c*np.sqrt((y*y) + 1.0) - total_impl_variance
        
        return (diff*diff).sum()
        
        

    def calibrate_inner(self, T , marketVols, strikes, F_0):
        

        log_moneyness = [np.log(x/F_0) for x in strikes]
        
    
        data = zip( T*np.ones(len(marketVols)) , F_0*np.ones(len(marketVols)) , marketVols , strikes , log_moneyness )
        self.data = data        
                
        new_y = np.array( [(i - self.m)/self.sigma for i in log_moneyness] )
        total_implied_variance = np.array([T * m*m for m in marketVols])

        
        y = new_y.sum()
        y2 = (new_y * new_y).sum()
        y2one = ((new_y*new_y) + 1.0).sum()
        ysqrt = np.sqrt((new_y*new_y) + 1.0).sum()
        y2sqrt = (new_y * np.sqrt((new_y*new_y) + 1.0) ).sum()
        
        v = total_implied_variance.sum()
        vy = (total_implied_variance * new_y).sum()
        vsqrt = (total_implied_variance * np.sqrt((new_y*new_y) + 1.0)).sum()
        
        matrix = [
                [1.0 , y , ysqrt] ,
                [y , y2 , y2sqrt], 
                [ysqrt, y2sqrt, y2one]
        ]
        A = np.array(matrix);   b = np.array([v,vy,vsqrt])
        
        a_, d_, c_ = np.linalg.solve(A,b)
        
        print "aHAT: {}, d: {}, c: {}, cost: {}".format(a_,d_,c_,np.sqrt(self.cost_function(new_y, a_, d_, c_, total_implied_variance)))
        
        if self.satisfy_inner_constraints(a_,d_,c_,total_implied_variance):
            
            self.ahat = a_
            self.d = d_
            self.c = c_
        
            self.a = self.ahat
            self.b = self.c / (self.sigma)                                          #Setting beta
            self.rho = self.d / (self.sigma * self.b)                               #Setting rho
        
        else:
            print "First Round not satisfied"
        
        ########### This shows that the condition is not satisfied.
        #now set the parameters back to None
        
        a_, d_, c_, cost_ = None, None, None, None
        count =1 
        
        for matrix, vector, clamp_params in [
                
                ([[1,0,0],[y,y2,y2sqrt],[ysqrt,y2sqrt,y2one]], [0, vy, vsqrt], False), # a = 0
                ([[1,0,0],[y,y2,y2sqrt],[ysqrt,y2sqrt,y2one]], [total_implied_variance.max(),vy,vsqrt], False), # a = _vT.max()
        
                ([[1,y,ysqrt],[0,-1,1],[ysqrt,y2sqrt,y2one]], [v,0,vsqrt], False), # d = c
                ([[1,y,ysqrt],[0,1,1],[ysqrt,y2sqrt,y2one]], [v,0,vsqrt], False), # d = -c 
        
                ([[1,y,ysqrt],[0,1,1],[ysqrt,y2sqrt,y2one]], [v,4*self.sigma,vsqrt], False), # d <= 4*s-c
                ([[1,y,ysqrt],[0,-1,1],[ysqrt,y2sqrt,y2one]], [v,4*self.sigma,vsqrt], False), # -d <= 4*s-c
        
                ([[1,y,ysqrt],[y,y2,y2sqrt],[0,0,1]], [v,vy,0], False), # c = 0
                ([[1,y,ysqrt],[y,y2,y2sqrt],[0,0,1]], [v,vy,4.0*self.sigma], False), # c = 4*S
        
                ([[1,y,ysqrt],[0,1,0],[0,0,1]], [v,0,0], True), # c = 0, implies d = 0, find optimal a
                ([[1,y,ysqrt],[0,1,0],[0,0,1]], [v,0,4.0*self.sigma ], True), # c = 4s, implied d = 0, find optimal a
        
                ([[1,0,0],[0,-1,1],[ysqrt,y2sqrt,y2one]], [0,0,vsqrt], True), # a = 0, d = c, find optimal c
                ([[1,0,0],[0,1,1],[ysqrt,y2sqrt,y2one]], [0,0,vsqrt], True), # a = 0, d = -c, find optimal c
        
                ([[1,0,0],[0,1,1],[ysqrt,y2sqrt,y2one]], [0,4*self.sigma,vsqrt], True), # a = 0, d = 4s-c, find optimal c
                ([[1,0,0],[0,-1,1],[ysqrt,y2sqrt,y2one]], [0,4.0*self.sigma,vsqrt], True) # a = 0, d = c-4s, find optimal c
            ]:                
                a, d, c = np.linalg.solve(np.array(matrix), np.array(vector))
                
                if clamp_params:
                    dmax = min(c, 4.0 * self.sigma - c)
                    a = min(max(a, 0), total_implied_variance.max())
                    d = min(max(d, -dmax), dmax)
                    c = min(max(c, 0), 4 * self.sigma)
                    
                cost = self.cost_function(new_y, a, d, c, total_implied_variance)
                
                print "count: {} , cost: {} , satisfied: {}".format(count, np.sqrt(cost), self.satisfy_inner_constraints(a,d,c,total_implied_variance))
                print "a: {}, d: {}, c: {}".format(a,d,c)
                print "\n"
                
                if self.satisfy_inner_constraints(a,d,c,total_implied_variance) and (cost_ is None or cost < cost_):
                    self.ahat, self.d, self.c, cost_ = a, d, c, cost
                                                    
                    self.a = self.ahat                     
                    self.b = self.c / (self.sigma)                                          #Setting beta
                    self.rho = self.d / (self.sigma * self.b)                               #Setting rho
                else:
                    #print "2nd rd not satisfied"
                    pass
                
                #print "a: {}, d: {}, c: {}".format(a,d,c)
                #print "count: {} , cost: {} , satisfied: {}".format(count, cost, self.satisfy_inner_constraints(a,d,c,total_implied_variance))
                count += 1
            
        
        print "\n\n FINAL aHat: {}, d: {}, c: {}".format(self.ahat,self.d,self.c)
        
    
    
    def calibrate_outer_now(self, T , marketVols, strikes, F_0):

        log_moneyness = [np.log(x/F_0) for x in strikes]
        lst = zip( T*np.ones(len(marketVols)) , F_0*np.ones(len(marketVols)) , marketVols , strikes , log_moneyness )
        
                        
        def objective_function_outer(w):
            
            self.m = w[0]
            self.sigma = w[1]
            
            return self.function_min_outer()
        
        
        initial_parameters = [self.m, self.sigma]
        
        min_logmoney = min(log_moneyness)
        max_logmoney = max(log_moneyness)
        
        min_sigma = 1e-12
        bnds = [(min_logmoney,max_logmoney) , (min_sigma,10.0)]
        
        result2 = differential_evolution(objective_function_outer,bounds= bnds)        
        #result2 = minimize(objective_function_outer,initial_parameters,method= "Nelder-Mead",options={'xtol':1e-15, 'disp':True} )
        #result2 = minimize(objective_function_outer,initial_parameters,args=(lst),method= "SLSQP",bounds=bnds,options={'ftol':1e-16, 'disp':True} )
        
        self.m = result2.x[0]
        self.sigma = result2.x[1]
        
        self.calibrated_ = True
        
        print result2
        
        
    
    
    ##################################        CALIBRATION METHOD 2         ###########################################################
    
    def function_minimise_inner_method2(self,data):
        
        T = data[0][0]                                      #timeslice at T
        marketVols = [z for (x,y,z,a,b) in data]            #market implied Vol
        log_moneyness = [b for (x,y,z,a,b) in data]         #get the list of log_moneyness
                         
        assert(len(marketVols) == len(log_moneyness))
        new_y = np.array( [(i - self.m)/self.sigma for i in log_moneyness] )
        
        cost = 0.0
        for i in range(len(marketVols)):
            cost += np.square(self.RawParametrization_Transformed(new_y[i]) - T*marketVols[i]*marketVols[i])
        
        return cost;

    
    def calibrate_new(self,T , marketVols, strikes, F_0):
        
        print "\n\n START OF CALIBRATION METHOD 2"                        
        
        log_moneyness = [np.log(x/F_0) for x in strikes]
    
        data = zip( T*np.ones(len(marketVols)) , F_0*np.ones(len(marketVols)) , marketVols , strikes , log_moneyness )
        self.data = data        
                        
        new_y = np.array( [(i - self.m)/self.sigma for i in log_moneyness] )
        total_implied_variance = np.array([T * m*m for m in marketVols])
        
        ########## Now solving for A,D,C ( INNER ) #############################
                        
        #first optimisation
        def objective_function_inner(x):
            
            self.ahat = x[0]
            self.c = x[1]
            self.d = x[2]
            
            return self.function_minimise_inner_method2(data)
        
        initial_parameters = [self.a, self.c, self.d]
        max_total_variance = max([T * m*m for m in marketVols])                                                 #MAX( observed total variance )
        
        cons1 = ({'type': 'ineq', 'fun': lambda x: (4.0*self.sigma) - x[1] },                                   # c <= 4*sigma
                {'type': 'ineq', 'fun': lambda x: x[1] - abs(x[2])  },                                          #|d| <= c
                {'type': 'ineq', 'fun': lambda x: (4.0*self.sigma - x[1]) - abs(x[2]) },                       #|d| <= 4*sigma - c
                {'type': 'ineq', 'fun': lambda x: max_total_variance - x[0] } )                                # a <= max(marketvols)


        cons2 = ({'type': 'ineq', 'fun': lambda x: (4.0*self.sigma) - x[1] },                                   # c <= 4*sigma
                {'type': 'ineq', 'fun': lambda x: x[2] + x[1]  },                                          #|d| <= c
                {'type': 'ineq', 'fun': lambda x: x[1] - x[2]  },
                {'type': 'ineq', 'fun': lambda x: (4.0*self.sigma) - x[1] - x[2] },                       #|d| <= 4*sigma - c
                {'type': 'ineq', 'fun': lambda x: x[2] - x[1] + (4.0*self.sigma) },
                {'type': 'ineq', 'fun': lambda x: max_total_variance - x[0] } )                                # a <= max(marketvols)


        bnds = [(0.0,1.0),(0.0,1.0),(-1.0,1.0)]
        #bnds = [(1e-8,10.0),(1e-8,10.0),(-10.0,10.0)]
        
        result = minimize(objective_function_inner,initial_parameters,method='SLSQP',bounds =bnds, constraints= cons1,options={'disp':True} )

        #print result
        
        self.ahat = result.x[0]                                                        #Setting a
        self.c = result.x[1]
        self.d = result.x[2]
        
        temp_rho = self.d / self.c
        
        self.a = self.ahat
        self.b = self.c / (self.sigma)                                          #Setting beta
        self.rho = self.d / (self.sigma * self.b)                               #Setting rho
        
        assert(temp_rho == self.rho)
                        
        print "\n Inner Calibration method 2 Done!"
        #print "a: {} , b: {} , rho: {}".format(self.a,self.b,self.rho)
        
        ########## Now solving for m and sigma ( OUTER ) #############################
        
        def objective_function_outer(x):
            
            self.m = x[0]
            self.sigma = x[1]
            
            return self.function_min_outer()
        
        
        initial_parameters = [self.m, self.sigma]
        
        min_logmoney = min(log_moneyness)
        max_logmoney = max(log_moneyness)
        
        min_sigma = 1e-12
        bnds = [(min_logmoney,max_logmoney) , (min_sigma,10.0)]
        
        result2 = differential_evolution(objective_function_outer,bounds= bnds)        
        #result2 = minimize(objective_function_outer,initial_parameters,method= "Nelder-Mead",options={'xtol':1e-15, 'disp':True} )
        #result2 = minimize(objective_function_outer,initial_parameters,method= "SLSQP",bounds=bnds,options={'ftol':1e-16, 'disp':True} )
        
        self.m = result2.x[0]
        self.sigma = result2.x[1]
        
        self.calibrated_ = True
        
        print result2
        

    

############################################# END ##########################################################################

    def calibration_report__(self):
        
        T = self.data[0][0]                                      #timeslice at T
        marketVols = [z for (x,y,z,a,b) in self.data]            #market implied Vol
        log_moneyness = [b for (x,y,z,a,b) in self.data]         #get the list of log_moneyness
        
        market_total_implied_variance = [(T*m*m) for m in marketVols]
                        
                        
        if self.calibrated_ == False:
            pass
        
        else:            
            model_total_implied_variance = [self.RawParametrization(X) for X in log_moneyness]
            model_vols = [np.sqrt(i/T) for i in model_total_implied_variance]
            
            calibration_report = pd.DataFrame(zip(market_total_implied_variance,model_total_implied_variance),columns=["Market Variance","Model Variance"], index=log_moneyness)
            #calibration_report = pd.DataFrame(zip(marketVols,model_vols),columns=["Market Implied Vol","Model Implied Vol"], index=log_moneyness)
        
            return calibration_report
                        


if __name__ == "__main__":
    
    #input_location = "C:\\Users\\Benedict Tan\\Dropbox\\Python\\Input\\"
    #input_location = "C:\\Users\\bened\\Dropbox\\Python\\Input\\"
    input_location = "H:\\python\\Input\\"
    
    #######################
    ###   Read in YC
    #######################
    
#    filename = "ois_rates"
#    cols = ["Date","Rate"]
#    df1 = pd.read_csv(input_location + "YieldCurve\\" + filename + ".csv",header =None, names = cols)
#    df1.Date = df1.Date.astype(int)
#    df1.Date = [ql.Date(d) for d in df1.Date]
#    today = ql.Date(30,ql.December,2016);
#    YC = TS.yieldcurve2(today,df1)
        
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
                                           
    #LocalVol.plot_impliedVolatilitySurface()                                
                                    
    ########################### CALIBRATION OF SVI ################################
    
#    LocalVol.calibrate_new();
#                                    
#    calibration_report = LocalVol.calibration_report__()
#    calibration_report.plot()                

    ####################     MARKET DATA      #####################################


    ###############################################################################

    #LocalVol.calibrate_inner(T , marketvols, strikes, S_0)
    #LocalVol.calibrate_outer_now()

    #calibration_report = LocalVol.calibration_report__()
    #calibration_report.plot()           


    #T = 0.021917808
    T = 1.0
    strikes = [12800,12850,12900,12950,13000,13050,13100,13150,13200,13250,13300,13350,13400,13450,13500,13550,13600,13650,13700,13750,13800,13850,13900,13950,14000]
    vol = [14.72676855,14.52676855,13.87728127,13.57728127,12.83483687,12.63483687,12.1159613,11.5,11.27042646,10.9120998,10.31113392,10.40121804,10.15033589,10.18864623,10.10272992,9.946587903,10.17992852,10.13286696,10.30465049,10.51437601,10.74636333,10.97605417,11.29326233,11.58377569,11.88377569]
    vol = [i/100 for i in vol]
    F_0 = 13369.68
    
    df = pd.read_csv("C:\\Users\\tanbened\\python\\Input\\SVI Interpolation Data\\spx_index_21sep.csv",header=None)
    
    strikes = df.iloc[:,0]
    vol = df.iloc[:,1]
    vol = [i/100 for i in vol]


    T = 0.693150685
    F_0 = 2753.31

    log_moneyness = [np.log(x/F_0) for x in strikes]
    plt.plot(log_moneyness, vol)

    LocalVol.calibrate_inner(T , vol, strikes, F_0)
    LocalVol.calibrate_outer_now(T , vol, strikes, F_0)
    
    LocalVol.get_paramS()    
    
    calibration_report = LocalVol.calibration_report__()
    calibration_report.plot()     

    ##################################################################################
    
#    LocalVol2 = SVI()
#    
#    LocalVol2.calibrate_new(T , vol, strikes, F_0)
#    LocalVol2.get_paramS()   
#
#    plt.figure()
#    calibration_report2 = LocalVol2.calibration_report__()
#    calibration_report2.plot()     
    
    
    
    
    
    




            
                                    
                                    
