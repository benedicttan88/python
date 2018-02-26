# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 10:01:10 2017
Make New Schedules
@author: WB512563
"""

from __future__ import division
import QuantLib as ql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def adjustToFirstCoupon(T):
    calendar = ql.WeekendsOnly();
    
    Nearest_Coupon_date = calendar.adjust(T + 1,ql.Following);
    if (T + 1 >= Nearest_Coupon_date):
        T = T + ql.Period(3,ql.Days);
    else:
        T = T - ql.Period(3,ql.Days);
                         
    return T;


def MakeNewCouponSchedule(start_date, tenors):
    calendar = ql.WeekendsOnly();
    
    if (start_date.dayOfMonth() == 20 or start_date.dayOfMonth() == 19):
        maturity = calendar.adjust(start_date + tenors, ql.Unadjusted)
    else:
        maturity = calendar.adjust(start_date + tenors, ql.Following)
                         
                                     
    #PreConditions:
    d1 = ql.Date(20,6,start_date.year())
    d2 = ql.Date(19,9,start_date.year())  
    
    startofyear = ql.Date(1,1,start_date.year())
    march19 = ql.Date(19,3,start_date.year());

    dec20 = ql.Date(20,12,start_date.year())
    endofthisyear = ql.Date(31,12,start_date.year())
    
    #Conditions:
    cond_setA_1 = (startofyear <= start_date < march19)
    cond_setA_2 = (d1 < start_date and start_date < d2);
    cond_setA_3 = (dec20 < start_date and start_date <= endofthisyear)
    
    cond_setB_1 = (start_date.dayOfMonth() == 20 and start_date.month() == 3)
    cond_setB_2 = (start_date.dayOfMonth() == 20 and start_date.month() == 6);
    cond_setB_3 = (start_date.dayOfMonth() == 20 and start_date.month() == 9);
    cond_setB_4 = (start_date.dayOfMonth() == 20 and start_date.month() == 12);

    cond_setC_1 = (start_date.dayOfMonth() == 19 and start_date.month() == 3);
    cond_setC_2 = (start_date.dayOfMonth() == 19 and start_date.month() == 6);
    cond_setC_3 = (start_date.dayOfMonth() == 19 and start_date.month() == 9);
    cond_setC_4 = (start_date.dayOfMonth() == 19 and start_date.month() == 12);

    #SET A (Only from 21-18 Every Quarterly)
    if (cond_setA_1 or cond_setA_2 or cond_setA_3):
        print "setA : " , start_date
    
        coupon_schedule = ql.Schedule(start_date, maturity - ql.Period(3,ql.Months),
                           ql.Period(ql.Quarterly),
                            calendar,
                            ql.Following,
                            ql.Unadjusted,
                            ql.DateGeneration.CDS,
                            False);
                                      
    #SET B (Only Days on the 20th Quarterly)
    elif (cond_setB_1):
        print "setB March: " , start_date
        start_date = adjustToFirstCoupon(start_date);
        
        coupon_schedule = ql.Schedule(start_date,maturity + ql.Period(3,ql.Days),
                           ql.Period(ql.Quarterly),
                            calendar,
                            ql.Following,
                            ql.Unadjusted,
                            ql.DateGeneration.CDS,
                            False);  
                                      
    elif (cond_setB_2):
        print "setB June: " , start_date
        start_date = adjustToFirstCoupon(start_date);
                 
        coupon_schedule = ql.Schedule(start_date,maturity - ql.Period(86,ql.Days),
                           ql.Period(ql.Quarterly),
                            calendar,
                            ql.Following,
                            ql.Unadjusted,
                            ql.DateGeneration.CDS,
                            False);
                                      
    elif (cond_setB_3):
        print "setB Sept: " , start_date
        start_date = adjustToFirstCoupon(start_date);
                 
        coupon_schedule = ql.Schedule(start_date,maturity + ql.Period(3,ql.Months) -ql.Period(3,ql.Days),
                           ql.Period(ql.Quarterly),
                            calendar,
                            ql.Following,
                            ql.Unadjusted,
                            ql.DateGeneration.CDS,
                            False);
                                      
    elif (cond_setB_4):
        print "setB Dec: " , start_date
        start_date = adjustToFirstCoupon(start_date);
                 
        coupon_schedule = ql.Schedule(start_date,maturity - ql.Period(3,ql.Days),
                           ql.Period(ql.Quarterly),
                            calendar,
                            ql.Following,
                            ql.Unadjusted,
                            ql.DateGeneration.CDS,
                            False);
                                      
    #SET C ( Only days on the 19th)
    elif (cond_setC_1):
        print "setC March: " , start_date
        start_date = adjustToFirstCoupon(start_date);
        
        coupon_schedule = ql.Schedule(start_date,maturity - ql.Period(3,ql.Months),
                           ql.Period(ql.Quarterly),
                            calendar,
                            ql.Following,
                            ql.Unadjusted,
                            ql.DateGeneration.CDS,
                            False);
    elif (cond_setC_2):
        print "setC June: " , start_date
        start_date = adjustToFirstCoupon(start_date);
   
        coupon_schedule = ql.Schedule(start_date + ql.Period(3,ql.Days),maturity - ql.Period(3,ql.Days),
                           ql.Period(ql.Quarterly),
                            calendar,
                            ql.Following,
                            ql.Unadjusted,
                            ql.DateGeneration.CDS,
                            False);
    elif (cond_setC_3):
        print "setC Sept: " , start_date     
        start_date = adjustToFirstCoupon(start_date);
   
        coupon_schedule = ql.Schedule(start_date, maturity - ql.Period(3,ql.Months),
                           ql.Period(ql.Quarterly),
                            calendar,
                            ql.Following,
                            ql.Unadjusted,
                            ql.DateGeneration.CDS,
                            False);                                       
    elif (cond_setC_4):
        print "setC Dec: " , start_date
        start_date = adjustToFirstCoupon(start_date);
 
        coupon_schedule = ql.Schedule(start_date + ql.Period(3,ql.Days),maturity - ql.Period(3,ql.Days),
                           ql.Period(ql.Quarterly),
                            calendar,
                            ql.Following,
                            ql.Unadjusted,
                            ql.DateGeneration.CDS,
                            False);
                                     
        #print list(enumerate(coupon_schedule))
        
    else:            
        coupon_schedule = ql.Schedule(start_date,maturity,
                           ql.Period(ql.Quarterly),
                            calendar,
                            ql.Following,
                            ql.Unadjusted,
                            ql.DateGeneration.CDS,
                            False);



    return coupon_schedule






if __name__ == "__main__":
    
    
    print "a"
    start_date = ql.Date(30,12,2016)
    tenors = ql.Period(1,ql.Years)
    
    coupon_schedule = MakeNewCouponSchedule(start_date,tenors)
    
    
    
    
    
    
    
    
    
