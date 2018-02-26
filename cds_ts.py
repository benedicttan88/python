# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 06:26:52 2017

@author: Benedict Tan
"""

import QuantLib as ql
import numpy as np
import pandas as pd



#location = "C:/Users/Benedict Tan/Dropbox/worldbank/hazard_rates_bootstrap/hazard_rates/"
location = "D:\\worldbank\\hazard_rates_bootstrap\\input\\"

df_ois = pd.read_csv(location + "OIS_USD_BSLIB_31032017.csv", header=None)
#df_ois = pd.read_csv(location + "ois_rates_31Mar.csv", header=None)
                    
def get_rates(location,filename):
              
    df_ois = pd.read_csv(location + filename + ".csv", header=None)

    rates = df_ois.iloc[:,1].tolist()
#    df = []
#    for i in range(len(spot_rates)):
#        df.append(np.exp(-spot_rates[i]*t[i]));

    return rates
    

def get_dates(location,filename):
    
    df_ois = pd.read_csv(location + filename + ".csv", header=None)
    
    ts_dates = [];
    for x in df_ois.iloc[:,0].astype(int):
        ts_dates.append(ql.Date(x))
                
    return ts_dates;
    



def make_dates(df):
    
    ts_dates = [];
    for x in df.iloc[:,0].astype(int):
        ts_dates.append(ql.Date(x))
        
    return ts_dates


def make_rates(today):
    
    spot_rates = (df_ois.iloc[:,4]/100).tolist()
    
    
    return spot_rates


if __name__ == "__main__":
    
    calendar =  ql.TARGET()
    today = ql.Date(31,ql.March,2017);
    Compounding = ql.Continuous
    interpolation = ql.Linear()
                   
    DayCounter2 = ql.Actual365Fixed();   
    ts_dates = get_dates(today);
    ts_rates = get_rates(today);
    
    tsCurve = ql.YieldTermStructureHandle(ql.ZeroCurve(ts_dates, ts_rates, DayCounter2, calendar, interpolation,
                                                       Compounding))                    
                        
    
    
    df_dates = pd.read_csv(location + "test_meng_IR.csv", header=None)
    dates = make_dates(df_dates)
    
    start_date = dates[0]
    
    adjustment_factor = 365/365.25 
    
    x = [tsCurve.discount((d - start_date)/365.25) for d in dates]
    y = [tsCurve.discount((d - start_date)/365.25)**adjustment_factor for d in dates]
    
    