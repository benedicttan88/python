# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 11:43:52 2017

@author: Benedict Tan
"""

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import MatrixDecompositions

class SABR(object):
    
    def __init__(self):
        
        self.a      = 0.01;
        self.beta   = 1
        self.nu     = 0.05
        self.rho    = -.04
        self.calibrated_ = False
        
    def set_alpha(self,a_):
        self.a = a_
    def set_beta(self,beta_):
        self.beta = beta_
    def set_nu(self,nu_):
        self.nu = nu_
    def set_rho(self,rho_):
        self.rho = rho_
    
    def printParams(self):
        print "a: {} , b: {} , nu: {} , rho: {}".format(self.a,self.beta,self.nu,self.rho)
        
    def get_Vol_hagan_LogNormalApproximation(self , K , F_0 , T):
        """
        K  Strike \n
        F_0 Forward Price at time 0 \n
        T  maturity of option \n
        """
        if self.rho == 1.0:
            self.rho -= 1.0 - 1e-8

        
        if F_0 != K:
        
            FK = F_0*K
            B = 1 - self.beta

            top = B*B*self.a*self.a/(24.0*np.power(FK,B)) + (self.rho*self.beta*self.nu*self.a)/(4*np.power(FK,0.5*B)) + self.nu*self.nu*(2.0-3*self.rho*self.rho)/24.0     
            top = self.a * (1 + top*T)                                    
            bottom = np.power(FK,0.5*B) * (1 + B*B*np.log(F_0/K)*np.log(F_0/K)/24.0 + np.power(B,4)*np.power(np.log(F_0/K),4)/1920.0 )
                                                                        
            z = self.nu * np.power(FK,0.5*B) * np.log(F_0/K) / self.a
                                  
            X_z_inside_top = ( np.sqrt(1 - 2*self.rho*z + z*z) + z - self.rho ) 
            X_z_inside_bot = 1 - self.rho
            X_z_inside = X_z_inside_top / X_z_inside_bot
            #print "XZ: ", X_z_inside
#            if X_z_inside == 1.0:
#                print "something here"
#                print X_z_inside_top
#                print "z:" , z
#                self.printParams()
#                #print "erea" , X_z_inside
#                #X_z_inside = X_z_inside + 1e-8
            if X_z_inside == 0:
                X_z_inside += 1e-9
            X_z = np.log( X_z_inside )
            
            if X_z == 0:
                print "here"
            
            return top * z / (bottom * X_z)
            
        else:   #ATM Option
        
            B = 1 - self.beta
            temp = B*B*self.a*self.a/(24.0*np.power(F,2*B)) + self.rho*self.beta*self.nu*self.a/(4*np.power(F,B)) + self.nu*self.nu*(2.0-3.0*self.rho*self.rho)/24.0 
            
            return self.a * (1 + temp*T) / np.power(F,B)
        

    

    def get_Vol_hagan_NormalApproximation(self, K , F_0 , T ):
        
        pass



    def function_to_minimise(self,lst,marketvols,strikes):
                
        #should change this
        F_0 = lst[0]
        T = lst[1]
        
        sum_ = 0.0
        
        for i in range(len(marketvols)):
            sum_ += np.square( marketvols[i] - self.get_Vol_hagan_LogNormalApproximation(strikes[i] , F_0 , T)  )
            
        return sum_
    
    
    def calibrate(self, T, F_0 , marketvols, strikes , ATM_vol , ATM_strike ):
        """
        calibration by fixing beta
        Calibrating to Swaptions now
        """
        self.calibrated_ = False
        print "\t\t Start of Calibration Process"
        
        
        lst = [F_0,T]
        error = 10
        counter = 0
        
        conditionList = []
        stopping_condition = 1e-10
        while error >= 1e-5:
            
            print "iteration {}".format(counter)

            #Getting the Roots of A
            a_roots = self.getRootAlpha(F_0,T, ATM_vol )
            positive_roots = [i for i in a_roots if i >=0]
            self.a = min(positive_roots)
            
            #Getting p and v given what is A
            def objective_function(x,lst,marketvols, strikes):
                
                self.nu = x[0]
                self.rho = x[1]
            
                return self.function_to_minimise(lst , marketvols , strikes)
            
            bnds = ((1e-8,1000),(-1,1))
            result = minimize(objective_function,[self.nu,self.rho],args=(lst,marketvols,strikes),method='SLSQP',bounds =bnds,options={'disp':True} )
            
            
            self.nu = result.x[0]
            self.rho = result.x[1]
             
            error = result.fun
            counter += 1
            
            conditionList.append(result.fun)
            if len(conditionList) > 2:
                conditionList = conditionList[1:]
                relative_error = conditionList[-1] - conditionList[0]
                if relative_error <= stopping_condition:
                    self.calibrated_ = True
                    break
                
            print conditionList
            if counter == 100:
                break
        
        
        return 0
        

    def getRootAlpha(self , F_0 , T , sigma_ATM ):
        
        #For ATM Option get cubic equation for a
        b = 1 - self.beta
        
        A = b*b*T/(24.0*np.power(F_0,2*b))
        B = self.rho*self.beta*self.nu*T/(4*np.power(F_0,b))
        C = 1 + (self.nu*self.nu*T*(2.0 - 3.0*self.rho*self.rho)/24.0)
        residual = -sigma_ATM*np.power(F_0,b)
        #getting roots for a
        coeffs = [A,B,C,residual]
        
        alpha_roots = np.roots(coeffs)
        print alpha_roots
        
        return alpha_roots
        
    
    
    def calibration_report(self, maturityT , F_0, marketVols, strikes_lst):
        
        if self.calibrated_ == False:
            
            pass
        else:
            
            modelVols = []
            
            for i in range(len(strikes_lst)):
                modelVols.append(self.get_Vol_hagan_LogNormalApproximation(strikes_lst[i],F_0,maturityT))
            
            calibration_report = pd.DataFrame(zip(marketVols,modelVols),columns=["Market Vols","Model Vols"],index=strikes_lst)
        
            print "SABR Calibrated Parameters: a: {} , B: {} , nu: {} , rho: {}".format(self.a,self.beta,self.nu,self.rho)
            
            return calibration_report
        
        
    
    def generate_paths(self,NumSimulations, NumSteps, Expiry, F_0):
        """
        Doing only Euler Discretisation with zero absorbing boundary \n
        No Antiethic Variates \n
        Number of Simulations: \n
        Number of TimeSteps: \n
        Expiry: \n
        Initial value of F_0: \n
        
        """
            
        dt = float(Expiry) / float(NumSteps)
        dt_sqrt = np.sqrt(dt)
        
        pathContainer = np.zeros((NumSimulations,NumSteps + 1) )
        pathContainer[:,0] = F_0                                 #Fill up the first column with initial value of F_0;

                     
        Updating_F = np.copy(pathContainer[:,0])
        Updating_a = np.ones(NumSimulations)*self.a
        
        correlationMatrix = np.array([[1.0, self.rho] , [self.rho , 1]])
        MatrixDecompositions.draw_N_randomNumbers(correlationMatrix)
        
                  
        for i in range(1,NumSteps + 1):
    
            for j in range(NumSimulations):

                #print "time: {} , Simulation: {}".format(i,j)
                #print "pathContainer j: {} , i: {}".format(j,i)
                
#                if ((self.beta > 0 and self.beta < 1) and F_t <= 0):
#                    F_t = 0
#                    pathContainer[i,j] = F_t
#                else:
#                    rand = MatrixDecompositions.draw_N_randomNumbers(correlationMatrix)
#                    dW_F = dt_sqrt * rand[0]
#                    
#                    Updating_F[j + 1] = Updating_F[j] + ( Updating_a[j] * np.power(abs(Updating_F[j]),self.beta) * dW_F)
#                    pathContainer[i,j] = Updating_F[j + 1]
#                    
#                    dW_a = dt_sqrt * rand[1]
#                    Updating_a[j + 1] = Updating_a[j] + (self.nu*Updating_a[j]*dW_a)
                
                rand = MatrixDecompositions.draw_N_randomNumbers(correlationMatrix)
                dW_F = dt_sqrt * rand[0]
                
                old_Fj = Updating_F[j]
                Updating_F[j] = Updating_F[j] + ( Updating_a[j] * np.power(abs(Updating_F[j]),self.beta) * dW_F)
                probability = np.exp( -2.0*old_Fj*Updating_F[j] / (Updating_a[j]*Updating_a[j]*np.power(Updating_F[j],2*self.beta)*dt)        )
                if ( (Updating_F[j] > 0.0) and (np.random.uniform(0,1) > probability) ):
                    pathContainer[j,i] = Updating_F[j]
                else:
                    pathContainer[j,i] = 0.0
                    
                dW_a = dt_sqrt * rand[1]
                Updating_a[j] = Updating_a[j]*np.exp(self.nu*dt_sqrt*dW_a - 0.5*self.nu*self.nu*dt)
                
                
        return pathContainer
               

if __name__ == "__main__":
    
    sabr = SABR();
    
    sabr.a = 0.03
    sabr.beta = 1.0
    sabr.nu = 0.9
    sabr.rho = -1
    
    
    ######### TEST SIGMA APPROXIMIATION ################
    
    maturity = 3.5
    F = 0.03571
    K = 0.5
    
    vol = sabr.get_Vol_hagan_LogNormalApproximation(K, F, maturity)
    
    print vol
    
    
    ######### CALIBRATION AREA #############
    
    marketvols = [26.97, 20.16, 17.63, 16.11 , 14.8, 14.66, 14.99]
    strikes = [2.597, 3.597, 4.097, 4.597, 5.097, 5.597, 6.597]
    
    marketvols = [x/100 for x in marketvols]
    strikes = [x/100 for x in strikes]
    marketvol_ATM = marketvols[3]
    strikes_ATM = strikes[3]
    
    maturity = 2
    F = strikes_ATM
    
    
    sabr.calibrate(maturity,F , marketvols , strikes , marketvol_ATM, strikes_ATM )
    calibration_report = sabr.calibration_report(maturity ,  F , marketvols , strikes)
    calibration_report.plot()
    
    
    ########################################

    NumberofSimulations = 300
    NumberofTimeSteps = 300
    Expiry = 60.0
    
    paths = sabr.generate_paths(NumberofSimulations , NumberofTimeSteps, Expiry , F)
    plt.figure()
    plt.hist(paths[:,-1])
    print "Expected Value F_T: {}".format(np.mean(paths[:,-1]))
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
        