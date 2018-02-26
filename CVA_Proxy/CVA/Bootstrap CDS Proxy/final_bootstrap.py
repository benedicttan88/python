# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:59:16 2017

@author: WB512563
"""

from __future__ import division
import QuantLib as ql
import numpy as np
import scipy.optimize
import math
import TS
import pandas as pd


def intersection(l1, l2):
    #l1 is the shorter list
    #l2 is the longer list
    
    length = np.minimum(len(l1),len(l2))    
        
    return l2[:length] , l2[length:]


def boot(start_date,Recovery,tsCurve,Quoted_Spreads,Schedule,DayCounter):
    
    print "Bootstrapping.."
    ######################################################
    ###     Settings
    ######################################################        
    
    df_full = pd.DataFrame(columns = ["CouponDates","Hazard"]);
    full_coupon_schedule_list = []
    Hazard_List = []

    maturity_dates = [start_date]
    for i in range(len(Schedule)):
        maturity_dates.append(list(enumerate(Schedule[i]))[-1][1])

    ######################################################
    ###     Looping
    ######################################################
    
    for i in range(len(Schedule)):

        quoted_spread = Quoted_Spreads[i];
        print "i = {} :  Schedule {}".format(i,i);
        print "quoted spread: ", quoted_spread   
        
        schedule_dates = []
        for d in enumerate(Schedule[i]):                               #get dates
            schedule_dates.append(d[1])
        schedule_dates[0] = start_date;
        
                      
        difference_coupon_schedules = intersection(full_coupon_schedule_list,schedule_dates)[1]

        
        def f(hazardRates, Recovery, quoted_spread, schedule_dates, tsCurve):
            
            
            sum_FLOAT = 0.0;
            sum_FIXED = 0.0;
            for j in range(1,len(schedule_dates)):
                
                mid_date = schedule_dates[j-1] + int(math.ceil(0.5*(schedule_dates[j] - schedule_dates[j-1])))
                df_mid = tsCurve.discount( mid_date )                                
                df = tsCurve.discount( schedule_dates[j] )
                A_j = (schedule_dates[j] - schedule_dates[j-1]) / 360;
             
                sum_FLOAT += df_mid * ( int_get_Survival_Probability(hazardRates,schedule_dates[j-1],Schedule,maturity_dates,Hazard_List,df_full) - int_get_Survival_Probability(hazardRates,schedule_dates[j],Schedule,maturity_dates,Hazard_List,df_full) )
                left = df * int_get_Survival_Probability(hazardRates,schedule_dates[j],Schedule,maturity_dates,Hazard_List,df_full)
                right = 0.5 * df_mid * ( int_get_Survival_Probability(hazardRates,schedule_dates[j-1],Schedule,maturity_dates,Hazard_List,df_full) - int_get_Survival_Probability(hazardRates,schedule_dates[j],Schedule,maturity_dates,Hazard_List,df_full) )
                sum_FIXED += A_j * (left + right)
                                                
                                
            CF_FLOAT = (1-Recovery) * sum_FLOAT
            CF_FIXED = quoted_spread * sum_FIXED;
            
            total = CF_FLOAT - CF_FIXED
            
            return total
        
        #new_hazard_rate = scipy.optimize.bisect(f, 0.000001, 0.05, args=(Recovery,quoted_spread,schedule_dates, tsCurve))
        new_hazard_rate = scipy.optimize.newton(f, 0.01, args=(Recovery,quoted_spread,schedule_dates, tsCurve), tol= 1e-11)
        
        
        #map the new hazard_rate ===> difference_ coupon_schedules
        b = np.empty(len(difference_coupon_schedules))
        b.fill(new_hazard_rate)
        C = pd.DataFrame(zip(difference_coupon_schedules, b),columns = ["CouponDates","Hazard"])
        df_full = df_full.append(C);
                                

        Hazard_List.append(new_hazard_rate)        
        full_coupon_schedule_list = schedule_dates
        
                        
        print "Bootstrapping Done"
        
    
    return Hazard_List


def getMaturityDates(start_date,Schedule_list):
    maturity_dates = [start_date]
    for i in range(len(Schedule_list)):
        maturity_dates.append(list(enumerate(Schedule_list[i]))[-1][1])
    
    return maturity_dates



def int_get_Survival_Probability(hazardRates,T,Schedule,maturity_dates,Hazard_List,df_full):
    
        
    #print maturity_dates
 
    ######################################################
    ###     Calculation
    ######################################################
    
    
    sums = 1.0;
    for i in range(1,len(maturity_dates)):
        
        if (T > maturity_dates[i]):
            sums *= np.exp(-Hazard_List[i-1] * (maturity_dates[i]-maturity_dates[i-1]) / 365 )
        else:
            if T in df_full["CouponDates"].tolist():
                sums *= np.exp( -np.float64(df_full.loc[df_full["CouponDates"] == T,"Hazard"]) * (T - maturity_dates[i-1]) / 365 )
                break
            else:
                sums *=  np.exp(-hazardRates * (T - maturity_dates[i-1]) / 365 )
                break;
            
    return sums;    





def get_Survival_Probability(T,start_date,Schedule,Hazard_List):
    
    
    ######################################################
    ###     Settings
    ######################################################                      
    
    
    maturity_dates = [start_date]
    for i in range(len(Schedule)):
        maturity_dates.append(list(enumerate(Schedule[i]))[-1][1])
    
    #print maturity_dates
    ######################################################
    ###     Calculation
    ######################################################
    
    
    sums = 1.0;
    for i in range(1,len(maturity_dates)):
        
        if (T > maturity_dates[i]):
            sums *= np.exp( -Hazard_List[i-1] * (maturity_dates[i]-maturity_dates[i-1]) / 365 )
        else:
            sums *=  np.exp( -Hazard_List[i-1] * (T - maturity_dates[i-1]) / 365 )
            break;
            
    return sums;    














