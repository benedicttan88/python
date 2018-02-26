# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 13:57:05 2017

@author: Benedict Tan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Settings
import TS
import calibration_hw1f

import QuantLib as ql



class HW1F_Constant_PathGeneration(object):
    
    def __init__(self,HW1F_obj_, YC_, temp_NumSimulations = 1000, t_NumDaysinYear = 365, temp_scheme = "Euler", temp_antiethic = True):
        
        self.a = HW1F_obj_.a
        self.sigma = HW1F_obj_.sigma
        self.YC = YC_
        
        self.NumSimulations = temp_NumSimulations
        self.NumDaysinYear = t_NumDaysinYear
        
        self.scheme = temp_scheme
        self.anthethic = temp_antiethic
        
    
#    def theta(self,t):             #Meng Theta
#        
#        deltaT = 1.0 / self.NumDaysinYear
#        
#        Fwd0 = self.YC.get_r(t, t + deltaT)
#        Fwd1 = self.YC.get_r(t + deltaT , t + 2*deltaT )
#                
#        return ((Fwd1 - Fwd0)/ deltaT) + self.a*Fwd0 +  self.sigma*self.sigma*(1.0 - np.exp(-2*self.a*t))/(2.0*self.a)
        
    
    
    
    def theta(self,t):              #My Theta
        
        deltaT = 1.0 / self.NumDaysinYear
        
        Fwd0 = self.YC.get_r(0, t)
        Fwd1 = self.YC.get_r(0 , t + deltaT )

        return ((Fwd1 - Fwd0)/ deltaT) + self.a*Fwd0 +  self.sigma*self.sigma*(1.0 - np.exp(-2*self.a*t))/(2.0*self.a)
    

    
    def get_vol(self,t):
        
        pass
    
    
    

    def generate_paths(self, Expiry , antihetic = True):
        
        
        N_Time = Expiry * self.NumDaysinYear
        
        deltaT = 1.0 / self.NumDaysinYear
        deltaT_sqrt = np.sqrt(deltaT)
        
        pathContainer = np.zeros((self.NumSimulations, N_Time + 1))
        pathContainer[:,0] = self.YC.get_r(0,deltaT)
        
        Updating_r = np.copy(pathContainer[:,0])
        
        theta = np.zeros(N_Time + 1)
        ########## Making Theta
        for i in range(1,N_Time):
            t = i * deltaT
            theta[i] = self.theta(t)
        
        print "NTIME: {}".format(N_Time)
        
        rng = np.random.RandomState(52)
        
        if antihetic == True: 
            
            for i in range(1, N_Time + 1):
                
                for j in range(self.NumSimulations / 2):
                    
                    rand = rng.randn()
                    
                    #rng = np.random.RandomState(52)
                    
                    #print "j: {} , i: {}".format(j,i)
                    
                    index = 2 * j
                    #print "container[{},{}]".format(index,i)
                    Updating_r[index] = Updating_r[index] + (theta[i] - self.a*Updating_r[index])*deltaT + self.sigma*rand*deltaT_sqrt
                    pathContainer[index,i] = Updating_r[index]          
                    
                    
                    index = 2*j + 1
                    #print "container[{},{}]".format(index,i)
                    Updating_r[index] = Updating_r[index] + (theta[i] - self.a*Updating_r[index])*deltaT - self.sigma*rand*deltaT_sqrt
                    pathContainer[index,i] = Updating_r[index]

                #print ""
            
            print "Path Generation Done"
            return pathContainer
            
        else:
            
            
            for i in range(1, N_Time + 1):
                
                for j in range(self.NumSimulations):
                    
                    rand = np.random.randn()
                    
                    Updating_r[j] = Updating_r[j] + (theta[i] - self.a*Updating_r[j])*deltaT + self.sigma*rand*deltaT_sqrt
                    pathContainer[j,i] = Updating_r[j]   
                    
            
            return pathContainer
            

    
    def getTime(self,Expiry):
        
        N_Time = Expiry * self.NumDaysinYear
        deltaT = 1.0 / self.NumDaysinYear
        
        timeVector = []
        for i in range(N_Time + 1):
            timeVector.append(i * deltaT)
        
        return timeVector
        
    
    #####################  For Expected Values #########################
    def alpha(self,t):
        
        deltaT = 1.0 / self.NumDaysinYear
        Fwd = self.YC.get_r(t, t + deltaT)
        
        return Fwd + 0.5*np.power((self.sigma*(1 - np.exp(-self.a*t))/self.a),2)


    

    def expected_value(self,s,t):
        
        deltaT = 1.0 / self.NumDaysinYear
        r0 = self.YC.get_r(0,deltaT)
        
        return r0 * np.exp(-self.a*(t-s)) + self.alpha(t) - self.alpha(s)*np.exp(-self.a*(t-s))

    
    def variance(self,s,t):
        
        return self.sigma*self.sigma*(1 - np.exp(-2*self.a*(t-s))) / (2.0 * self.a)        
        
    
    



if __name__ == "__main__":
    
    
    location = "D:\\python\\Input\\"
    
    
    df = pd.read_csv(location + "Yieldcurve\\" + "ois_rates" + ".csv", header=None)
    cols = ["Date","Rate"]
    df.columns = cols
    df.Date = df.Date.astype(int)
    
    df.Date = [ql.Date(d) for d in df.Date]
    
    
    today = ql.Date(30,ql.December,2016)
    
    YC = TS.yieldcurve2(today,df);                                          
    
    HW = calibration_hw1f.HW1F_constant(YC)
    
    
        
    #######  Caplet vol location
    black_caplet_vols = pd.read_csv(location + "Caplet Vols\\" + "Caplet ATM Vol from Cap Object" + ".txt",delimiter='\t')
    
    mat=[]; vol =[]
    for index,row in black_caplet_vols.iterrows():
        mat.append(row[0])
        vol.append(row[1])
    
    black_vols = zip(mat,vol)
    
    
    HW.calibrate(black_vols)
    calibration_report = HW.calibration_report()
    calibration_report.plot()
            
    
    ######################### SIMULATION AREA    
    
    NumSimulations = 1000
    NumDaysinYear = 250
    Maturity = 30
    
    
    HW_PathGeneration = HW1F_Constant_PathGeneration(HW,YC, NumSimulations, NumDaysinYear)
    paths = HW_PathGeneration.generate_paths(Maturity)
    plt.figure()
    plt.hist(paths[:,-1])    
    
    timeVector = HW_PathGeneration.getTime(Maturity);

    paths = pd.DataFrame(paths, columns = timeVector)
    plt.figure()
    paths.mean().plot()
                                          

    paths.iloc[:10,:]




#    paths_NA = HW_PathGeneration.generate_paths(Maturity,False)
#    plt.figure()
#    plt.hist(paths_NA[:,-1])
#    
#    timeVector = HW_PathGeneration.getTime(Maturity);
#
#    paths_NA = pd.DataFrame(paths_NA, columns = timeVector)
#    plt.figure()
#    paths_NA.mean().plot()


    ########### Expected Value
    
    


#
#    lst = []
#    N_Time = Maturity * NumDaysinYear
#        
#    deltaT = 1.0 / NumDaysinYear
#    for i in range(N_Time + 1):
#        t = i * deltaT
#        lst.append(t)
#    
#    
#    
#    HWexpected = [HW_PathGeneration.expected_value(0,x) for x in lst]
#
#
#    
#    
#    plt.plot(lst,HWexpected)
#    










    