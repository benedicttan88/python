# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 10:15:29 2017

@author: WB512563
"""

import QuantLib as ql

import numpy as np
import pandas as pd

import cds_ts
import TS



#### General Settings
location = "D:\\worldbank\\hazard_rates_bootstrap\\input\\"
ois_location = "D:\\python\\Input\\Yieldcurve\\"
index_curve_location = "D:\\python\\Input\\Yieldcurve\\"

ois_file = "ois_df_30jun2017"
index_curve_file = "usd_lib6m_df_30jun2017"

############# QL Pricing of SWAP ########################################################


#General Settings ###########

#date = "30Jun2017"
today = ql.Date(30,ql.June,2017)
ql.Settings.instance().evaluationDate = today;
                 
Notional = 200000000;

                              
#Fixed Details ###############
fixed_rate = 2.4335/100;

fixed_leg_effective_date = ql.Date(15,ql.July,2017)
fixed_leg_maturity_date = ql.Date(15,ql.July,2040)

fixed_leg_day_count = ql.Thirty360();
fixed_leg_freq = ql.Semiannual;
fixed_leg_BusinessdayConvention = ql.Following;
fixed_leg_calendar = ql.TARGET();
fixed_leg_DateGenerationRule = ql.DateGeneration.Forward

fixed_leg_currency = "USD"

#Float Details ###############

float_spread = 5/10000;

float_leg_effective_date = ql.Date(15,ql.July,2017)
float_leg_maturity_date = ql.Date(15,ql.July,2040)


float_leg_day_count = ql.Actual360();
float_leg_freq = ql.Semiannual;
float_leg_BusinessdayConvention = ql.Following;
float_leg_calendar = ql.TARGET();
float_leg_DateGenerationRule = ql.DateGeneration.Forward

############################################################################################



##########################  #create term structure   # MY TERMSTRUCTURE ####################
#reload(TS)
#df1 = pd.read_csv(location + "ois_rates_" + date + ".csv", header=None)          #TS
#cols = ["Date","Rate"]
#df1.columns = cols
#df1.Date = df1.Date.astype(int)
#df1.Date = [ql.Date(d) for d in df1.Date]
#
#YC = TS.yieldcurve(today,df1);
############################################################################################
                  
                  
############################  Discounting OIS Curve ########################################

ts_dates = cds_ts.get_dates(ois_location,ois_file);
ts_rates = cds_ts.get_rates(ois_location,ois_file);
                           
#tsCurve = ql.YieldTermStructureHandle(ql.ZeroCurve(ts_dates, ts_rates, DayCounter, calendar))
tsCurve = ql.YieldTermStructureHandle(ql.DiscountCurve(ts_dates, ts_rates, float_leg_day_count, float_leg_calendar))



############################################################################################

############################  Libor Projection Curve #######################################

ts_dates = cds_ts.get_dates(index_curve_location,index_curve_file);
ts_rates = cds_ts.get_rates(index_curve_location,index_curve_file);

liborCurve = ql.YieldTermStructureHandle(ql.DiscountCurve(ts_dates, ts_rates, float_leg_day_count, float_leg_calendar))


#libor3M_index = ql.Euribor3M(libor_curve)  
libor6M_index = ql.USDLibor(ql.Period(6, ql.Months), liborCurve)

############################################################################################


############################  SCHEDULE #####################################################
#create fixed schedule
fixed_coupon_schedule = ql.Schedule(fixed_leg_effective_date, fixed_leg_maturity_date,
                       ql.Period(fixed_leg_freq),
                        fixed_leg_calendar,
                        fixed_leg_BusinessdayConvention,
                        ql.Unadjusted,
                        fixed_leg_DateGenerationRule,
                        False);

#create float schedule
float_coupon_schedule = ql.Schedule(float_leg_effective_date, float_leg_maturity_date,
                       ql.Period(float_leg_freq),
                        float_leg_calendar,
                        float_leg_BusinessdayConvention,
                        ql.Unadjusted,
                        float_leg_DateGenerationRule,
                        False);
#############################################################################################

Flavor = ql.VanillaSwap.Payer

ir_swap = ql.VanillaSwap(Flavor,Notional,fixed_coupon_schedule,fixed_rate,fixed_leg_day_count, 
                         float_coupon_schedule,libor6M_index,float_spread,float_leg_day_count)


swap_engine = ql.DiscountingSwapEngine(tsCurve)
ir_swap.setPricingEngine(swap_engine)




####  Display Details ############

for i, cf in enumerate(ir_swap.leg(0)):
    print "%2d    %-18s  %10.2f"%(i+1, cf.date(), cf.amount())

print "%-20s: %20.3f" % ("Net Present Value", ir_swap.NPV())
print "%-20s: %20.3f" % ("Fair Spread", ir_swap.fairSpread())
print "%-20s: %20.3f" % ("Fair Rate", ir_swap.fairRate())
print "%-20s: %20.3f" % ("Fixed Leg BPS", ir_swap.fixedLegBPS())
print "%-20s: %20.3f" % ("Floating Leg BPS", ir_swap.floatingLegBPS())




    
    













