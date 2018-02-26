# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 15:06:39 2017

@author: Benedict Tan
"""


from __future__ import division

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm

import TS
import QuantLib as ql

from scipy.optimize import minimize
from scipy.optimize import differential_evolution

class Caplet(object):
    
    """
    Need to construct YC before pricing caplet
    """
    
    def __init__(self,YC_, t_Notional):
        #self.type;
        self.YC = YC_
        
        self.Notional = t_Notional
        
        self.a = 0.002
        self.b = 0.024
        self.c = 0.02
        self.d = 0.01
        
    
        
    def input_ATM_vols(self,location,filename):
        """
        has to be a .csv file
        """
        temp_df = pd.read_csv(location + filename + ".csv")
        self.data = temp_df
        
    def input_VolSurface(self,location,filename):
        """
        has to be a .csv file
        """
        temp_df = pd.read_csv(location + filename + ".csv")
        self.data = temp_df
    
        
        
    def price_black_caplet(self,Notional,vol,df,t,T,Fwd,K):
        """
        Fwd = Forward Rate at t \n
        K = Strike \n
        Notional = Notional \n
        t = time t \n
        T = Caplet maturity \n
        vol = Caplet vol
        """

        d1_ = (np.log(Fwd/K) + 0.5*vol*vol*(T - t)) / (vol * np.sqrt(T - t)) 
        d2_ = d1_ - (vol*np.sqrt(T - t))
        
        
        return self.YC.discount(t,T) * Notional * tau * (Fwd*norm.cdf(d1_) - K*norm.cdf(d2_))

    #############################################################################################################################

    def caplet_mv(self, vol, Notional ,t , T1 , T2, K):
        
        #lognormal black
        df = self.YC.discount(0,T2)
        Fwd = self.YC.forward(T1,T2)
        
        tau = T2 - T1
        
        d1_ = (np.log(Fwd/K) + 0.5*vol*vol*(T1 - t)) / (vol * np.sqrt(T1 - t)) 
        d2_ = d1_ - (vol*np.sqrt(T1 - t))
        
        #print d1_
        #print d2_
        
        return df*Notional*tau*(Fwd*norm.cdf(d1_) - K*norm.cdf(d2_))

    def cap_mv(self, cap_flat_vol , K , freq , cap_end):
        
        """"
        Get the Market Price of Cap from T1 to T2
        K = strike
        """
        t = 0
        T2 = cap_end
        counter = freq          #start = freq 
        
        #print "T2: {}".format(T2)
        sum_ = 0.0
        while counter < T2:

            T1 = counter
            counter = counter + freq
            #print "T1: {} , T2: {}".format(T1,counter)
                
            sum_ += self.caplet_mv(cap_flat_vol , self.Notional , t , T1 , counter , K )   #FIX THIS ONE FIRST
        
        return sum_
    
    def cap_mv_ABCD(self, a , b , c , d, K , freq , cap_end):
        """
        a , b , c , d \n
        K = strike \n
        freq = frequency of cap \n
        cap_end = T2 maturity of cap \n
        """
        T2 = cap_end
        start = freq
        
        sum_ = 0.0
        while start < T2:

            T1 = start
            start = start + freq
            #print "T1: {} , T2: {}".format(T1,start)
            spot_vol = self.ABCD_spot_vol(t, t, t , a , b , c , d)
            sum_ += self.caplet_mv(spot_vol , self.Notional , t , T1 , start , K )   #FIX THIS ONE FIRST
        
        return sum_
    
    ###################################################################################################################################

        
    def function_to_minimise_caplets(self):
        """
        ATM Cap Vols
        Calibration to caplets here
        Objective function: cap_market_values     
        weights = 1/N
        vol interpolation:      ABCD
        """
        
        freq =  0.5
        
        T2 = [1.0,2,3,4,5,6,7,8,9,10,12,15,20,25,30.0]
        market_cap_vols = [25.28, 30.55, 31.99, 34.11, 35.09, 35.39, 37.04, 35.11, 34.49, 34.54, 32.89, 30.82, 28.69, 27.61, 26.9]
        market_cap_vols = [x/100 for x in market_cap_vols]
        
        
        market_cap_prices = []
        for i in range(len(market_cap_vols)):
            t = 0
            
            #print "i: {}".format(i)
            K = self.YC.get_swap_rate(t, freq , T2[i] , freq)
            market_cap_price = self.cap_mv(market_cap_vols[i] , K , freq , T2[i])
            market_cap_prices.append(market_cap_price)
        
        sum_ = 0.0
        for i in range(len(market_cap_vols)):
            K = self.YC.get_swap_rate(t, freq , T2[i] , freq)            
            model_cap_price = self.cap_mv_ABCD(self.a , self.b , self.c , self.d , K , freq , T2[i])
            
            sum_ += abs(market_cap_prices[i] - model_cap_price)/ market_cap_prices[i] 
    
        return sum_
    
    
    
    
    def bootstrap_ABCD_Caplets(self):
        """
        For ATM Cap Vols
        """
        def objective_function(x):
            
            self.a = x[0]
            self.b = x[1]
            self.c = x[2]
            self.d = x[3]
            
            return self.function_to_minimise_caplets()
        
        initial_parameters = [self.a, self.b, self.c, self.d]
        bnds = ((-7,7),(-7,7),(1e-2,7),(1e-2,7))
        
        result = differential_evolution(objective_function,bounds= bnds)    
        
        self.a = result.x[0]
        self.b = result.x[1]
        self.c = result.x[2]
        self.d = result.x[3]
        


    def bootstrapping_ATM_PiecewiseLinear(YC, Tenor_list, Volatility_List):
        
        Tau = np.diff(Tenor_List)/365.0;
        Notional = 10000;
        
                     
                     
                     
        freq = 0.5
        for i in range(len(Tenors) - 1):
            
            ATM_strike = YC.get_swap_rate();
            ATM_strike_next = YC.get_swap_rate();
    
    
            if (i==1):
                ATM_Strike = YC.get_swap_rate();
                df = YC.discount(Tenor_List[0]);
                Fwd = YC.getFwd();
                black_mkt_price = black(Volatility_list[0],df,Notional,Tau[0],0,Tenor_list[0],Fwd,ATM_Strike)
                #black mkt price(vol,T=0,T=1) = Caplet_Price(vol1,T=0.5,T=1.0)
                #solve for this vol
            else:
                
                #Assume that each Cap is separated by 1 year apart
                caplet_prices = black(cap_vol[i+1],df1, Notional,tau1,0, T[i+1], YC.forward(T1,T2), cap_strike[i+1]) -\
                                     black((cap_vol[i],df2, Notional,tau2,0, T[i], YC.forward(T1,T2), cap_strike[i+1]))
            
            #LHS
            #sum of caplet prices will equal RHS
            
    #        if (Tenors[i] <1 ):
    #            print   " "
    #        else:
    #            RHS = black(caplet_vol,df1,Notional,tau,) +  black(caplet_vol,...)
    #            equate RHS with LHS and solve using optimsiation formula
                
        return Trnors
        
        
        
    
    
    ####################################### ABCD AREA ######################################################
    
    def ABCD_params(self):
        print "a: {} , b: {} , c: {} , d: {}".format(self.a,self.b,self.c,self.d)
    
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


    #########################################################################################################


    def function_to_minimise(self):
        """
        Calibration of Cap Vol to ABCD Model Vol
        
        """
        t = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        market_vols = [18.03, 19.15, 18.62, 17.73, 16.79, 15.81, 15.27, 14.87, 14.47, 14.13, 13.8, 13.47, 13.14, 12.81, 12.71, 12.68, 12.65, 12.63, 12.6]
        market_vols = [i/100 for i in market_vols]
        
        sum_ = 0.0
        for i in range(len(market_vols)):
            left = self.ABCD_I(t[i], t[i], t[i], self.a, self.b, self.c, self.d)
            right = self.ABCD_I(0.0, t[i] , t[i] , self.a, self.b, self.c, self.d)
            temp = self.ABCD_I(t[i], t[i], t[i], self.a, self.b, self.c, self.d) - self.ABCD_I(0.0, t[i] , t[i] , self.a, self.b, self.c, self.d)
            if temp <= 0.0:
                print "t = {}".format(t[i])
                print "a: {} , b: {} , c: {} , d:{}".format(self.a,self.b,self.c,self.d)
                print "left: {}  right: {}".format(left,right)
                print "temp: {}".format(temp)
            model_vol = np.sqrt( temp / t[i] )
            #sum_ += abs( (market_vols[i] - model_vol) / market_vols[i] )
            sum_ += abs((market_vols[i] - model_vol))
            
            penalty = self.penalty();
            lambda_ = 0.0
        return sum_ + lambda_*penalty

    def penalty(self):
        
        return np.square(max(0,(-self.a - self.d)))

    def bootstrap_ABCD(self):
    
        def objective_function(x):
            
            self.a = x[0]
            self.b = x[1]
            self.c = x[2]
            self.d = x[3]
            
            return self.function_to_minimise()
        
        initial_parameters = [self.a, self.b, self.c, self.d]
        #cons = ({'type': 'ineq', 'fun': lambda x: ( x[0] + x[3] ) })
        bnds = ((-7,7),(-7,7),(1e-2,7),(1e-2,7))
        
        #result = minimize(objective_function,initial_parameters,method='SLSQP',constraints= cons, bounds =bnds, options={'disp':True} )
        result = differential_evolution(objective_function,bounds= bnds)
        
        self.a = result.x[0]
        self.b = result.x[1]
        self.c = result.x[2]
        self.d = result.x[3]
        
        

    def calibration_report(self):
        
        t = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        market_vols = [18.03, 19.15, 18.62, 17.73, 16.79, 15.81, 15.27, 14.87, 14.47, 14.13, 13.8, 13.47, 13.14, 12.81, 12.71, 12.68, 12.65, 12.63, 12.6]
        market_vols = [i/100 for i in market_vols]
    
        model_vols = []
        for i in range(len(market_vols)):
        
            temp = self.ABCD_I(t[i], t[i], t[i], self.a, self.b, self.c, self.d) - self.ABCD_I(0.0, t[i] , t[i] , self.a, self.b, self.c, self.d)
            vol = np.sqrt( temp / t[i] )
            model_vols.append(vol)
            
            
        calibration_report = pd.DataFrame(zip(market_vols,model_vols),columns=["Market Vol","Model Vol"],index=t)
        
        return calibration_report




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
    
    Tenor_List = data.Tenor.tolist()
    Volatility_list = data.Volatility.tolist()
    
    
    
    df = 0.99642
    Notional = 1000000
    tau = 0.5167
    
    Fwd = 1.0125/100
    K = 2/100
    
    t = 0.0
    T1 = 0.5096
    T2 = 1.0
    caplet_vol = 51.29/100
    
    
#    print (black(caplet_vol,df,Notional,tau,t,T1,T2,Fwd,K))
#    
#
#    def wrapper(caplet_vol,df, Notional,tau, t,T1,T2,Fwd, K):
#        mkt_price = 32.52    
#        return mkt_price - black(caplet_vol,df, Notional,tau, t,T1,T2,Fwd, K)
#    
#    
#    print ( scipy.optimize.newton(wrapper,0.7,args=(df,Notional,tau,t,T1,T2,Fwd,K)) );
#          
#          
    
    
    #################Bootstrapping Cap Vol ABCD
     
    caplet = Caplet(YC,Notional)
    caplet.bootstrap_ABCD()
    
    calibration_report = caplet.calibration_report()
    calibration_report.plot()
    
    
    ##################################################
    
    #caplet.bootstrap_ABCD_Caplets()
    #caplet.function_to_minimise_caplets()
    