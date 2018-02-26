# -*- coding: utf-8 -*-
"""
Created on Wed May 03 10:25:20 2017

@author: WB512563
"""

from __future__ import division
import pandas as pd
import numpy as np
import scipy.optimize
import QuantLib as ql
import matplotlib.pyplot as plt
from collections import OrderedDict


#import ben_Bootstrap2
import CDS_SingleName_Schedules
import final_bootstrap
#import Bootstrap
#import cds_ts

import TS

import misc

################################################
####            Data Input
################################################

today = ql.Date(31,ql.March,2017)
date = "30Jun2017"


#Data
location = "D:\\python\\CVA_Proxy\\Input\\"
output_location = "D:\\python\\CVA_Proxy\\Output\\CDSBootstrap\\"

df = pd.read_excel(location + "Model2+NS_Govt" + ".xlsx")      #CDS
df1 = pd.read_csv(location + "ois_rates_31Mar2017" + ".csv", header=None)          #TS

                 
Tickers = df.Ticker.tolist()
termstructurelist = df.columns[1:12].str.upper().str[6:].tolist()


################################################
####            Term Structure
################################################
## TS_NEW
reload(TS)
cols = ["Date","Rate"]
df1.columns = cols
df1.Date = df1.Date.astype(int)
df1.Date = [ql.Date(d) for d in df1.Date]

YC = TS.yieldcurve(today,df1);

                  
spread_list = pd.DataFrame()                  
#Trade Details (Quoted Spreads)
count = 0
for index,row in df.iterrows():
    
    print "Ticker: {}".format(row["Ticker"])
    
    recovery_rate = row["Recovery"]
    quoted_spreads = row["Spread6m":"Spread30y"]
                
    quoted_spreads = (quoted_spreads.astype(str).str[:-1].astype(float))
    quoted_spreads = np.array(quoted_spreads)    
            
        
    ################################################
    ####            Settings
    ################################################
    
    calendar = ql.WeekendsOnly();
    today = calendar.adjust(today)
    ql.Settings.instance().setevaluationDate = today
    DayCounter = ql.Actual360();
    settlementDays = 0;             

    #create tenors for CDS Spreads
    tenors = [];
    for i in range(len(termstructurelist)):
        if (termstructurelist[i][-1] == 'M'):
            n = int(termstructurelist[i][:-1])
            tenors.append(ql.Period(n, ql.Months))
        elif (termstructurelist[i][-1] == 'Y'):
            n = int(termstructurelist[i][:-1])
            tenors.append(ql.Period(n, ql.Years))


    ################################################
    ####            Creating New Schedule
    ################################################
    
    #IPV - Schedules(NEW SCHEDULE)
    schedule_list2 = [];
    for i in range(len(tenors)):
        coupon_schedule = CDS_SingleName_Schedules.MakeNewCouponSchedule(today,tenors[i]);
        schedule_list2.append(coupon_schedule)
                                              
    schedule = schedule_list2;

    #print coupon schedule to excel file
    #misc.to_excel(schedule,output_location + "_" + country + ".xlsx");
             
    #print list of discount factors to excel file

    #General Settings
    maturity_dates = final_bootstrap.getMaturityDates(today,schedule)

#disc_dates = [d for i,d in enumerate(schedule[-1])]
#disc_dates[0] = ql.Date(31,ql.March,2017)
#D = [YC.discount(d) for d in disc_dates]
#serial = [d.serialNumber() for d in disc_dates]
#
#Disc_Fac = pd.DataFrame(serial,D).T
#Hazaraw_input(" ");
         
    ################################################
    ####            Short Bootstrapping
    ################################################
    reload(final_bootstrap)
    Hazard_List = final_bootstrap.boot(today,recovery_rate,YC,quoted_spreads,schedule,DayCounter)   
    Survival_Rates = [final_bootstrap.get_Survival_Probability(t,today,schedule,Hazard_List) for t in maturity_dates]
    
    
    
    Hazard_List = [0.0] + Hazard_List
    result = pd.DataFrame(zip([d.serialNumber() for d in maturity_dates],Hazard_List,Survival_Rates),columns = ["Date","Hazard","Survival"])

    spread_list = pd.concat([spread_list,result.Hazard],axis=1)
    count = count + 1

spread_list.columns = Tickers
columns = [" "] + termstructurelist
#columnnames = columnnames.append(pd.Index(["Recovery"]))
spread_list.index = columns


