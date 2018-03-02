# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:24:05 2017

@author: WB512563
"""
from __future__ import division
import QuantLib as ql
import numpy as np
import pandas as pd

import TS


#Bootstrapping Caplet QuantLib

####### location

location = "D:\\python\\Input\\"
filename = "atm_cap_vol"

df = pd.read_csv(location + filename + ".csv")


#####################################
####     SET UP YIELDCURVE
filename = "ois_rates"
cols = ["Date","Rate"]
df1 = pd.read_csv(location + "YieldCurve\\" + filename + ".csv",header =None, names = cols)
df1.Date = df1.Date.astype(int)
df1.Date = [ql.Date(d) for d in df1.Date]
today = ql.Date(30,ql.December,2017);
               
YC = TS.yieldcurve2(today,df1)

#####################################
####    QuantLib YIELDCURVE

YC_QuantLib = ql.ZeroCurve(df1.Date,df1.Rate, ql.Actual360(), ql.TARGET(), ql.Linear(), ql.Compounded, ql.Annual)
YC_QuantLib_Handle = ql.YieldTermStructureHandle(YC_QuantLib)




settlement_days = 2
calendar = ql.TARGET()
bdc = ql.ModifiedFollowing
daycounter = ql.Actual365Fixed()


cap_freq = 0.25
Strikes = [YC.get_ATM_Cap_Strike(0,cap_freq,x,cap_freq) for x in df.Expiry]
optionTenors = [ql.Period(int(i),ql.Years) for i in df.Expiry]
Vols_t = df.ATM.tolist()

Vols = ql.Matrix(len(optionTenors),1)
for i in range(len(optionTenors)):
    Vols[i][0] = Vols_t[i]/100


#vol curve == use vector
#vol surface ==  use matrix

capfloor_volsurface =  ql.CapFloorTermVolCurve(settlement_days, calendar, bdc, optionTenors, Vols_t, daycounter)    


ibor_index = ql.USDLibor(ql.Period(3, ql.Months), YC_QuantLib_Handle)

optionlet_surf = ql.OptionletStripper2(capfloor_volsurface, ibor_index)





#####################

#####################
filename = "CapFloor - USD"
fd = pd.read_csv(location + filename + ".csv",skiprows =3)

subset_df = fd.iloc[:,3:]
volMatrix = ql.Matrix(subset_df.shape[0],subset_df.shape[1])
for i in range(subset_df.shape[0]):
    for j in range(subset_df.shape[1]):
        volMatrix[i][j] = subset_df.iloc[i,j]/100


optionTenors = [ql.Period(int(i),ql.Years) for i in fd.iloc[:,0]]
Strikes = [0.25,0.5,0.75,1,1.5,2,3,3.5,4,5,6,7,8,9,10,11,12,13,14]
Strikes = [i/100 for i in Strikes]

ibor_index = ql.USDLibor(ql.Period(1, ql.Years), YC_QuantLib_Handle)

capfloor_volSurface = ql.CapFloorTermVolSurface(settlement_days,calendar,bdc,optionTenors,Strikes,volMatrix,daycounter)

optionlet_surf = ql.OptionletStripper1(capfloor_volSurface, ibor_index)










