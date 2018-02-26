# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:23:43 2017

@author: WB512563
"""

import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt


import math
from collections import namedtuple

##########################
###  Reading in File Data
##########################

input_location = "D:\\python\\Input\\Yieldcurve\\"





###########################
### Fixed Data for Examples
###########################

Swaption_CalibrationData = namedtuple("CalibrationData", "start, length, volatility")
swaption_data = [Swaption_CalibrationData(1, 5, 0.3077),
                 Swaption_CalibrationData(2, 4, 0.3224),
        Swaption_CalibrationData(3, 3, 0.3289),
        Swaption_CalibrationData(4, 2, 0.3328),
        Swaption_CalibrationData(5, 1, 0.3375 )]



                                
Cap_CalibrationData = namedtuple("CalibrationData", "length, volatility")
cap_data = [Cap_CalibrationData(1, 0.2006),
        Cap_CalibrationData(2, 0.3079),
        Cap_CalibrationData(3,0.3146),
        Cap_CalibrationData(4,0.3325),
        Cap_CalibrationData(5,0.3406 ),
        Cap_CalibrationData(6,0.3386 ),
        Cap_CalibrationData(7,0.3818 ),
        Cap_CalibrationData(8,0.3558 ),
        Cap_CalibrationData(9,0.3372 ),
        Cap_CalibrationData(10,0.3264 ),
        Cap_CalibrationData(12,0.3176 ),
        Cap_CalibrationData(15,0.3288 ),
        Cap_CalibrationData(20,0.3110 ),
        Cap_CalibrationData(25,0.2994 ),
        Cap_CalibrationData(30,0.2891 )]                   
            



##########################
###  Swaption Helper
##########################


def create_swaption_helpers(data, index, term_structure, engine):
    swaptions = []
    fixed_leg_tenor = ql.Period(1, ql.Years)
    fixed_leg_daycounter = ql.Actual360()
    floating_leg_daycounter = ql.Actual360()
    for d in data:
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(d.volatility))
        helper = ql.SwaptionHelper(ql.Period(d.start, ql.Years),
                                   ql.Period(d.length, ql.Years),
                                   vol_handle,
                                   index,
                                   fixed_leg_tenor,
                                   fixed_leg_daycounter,
                                   floating_leg_daycounter,
                                   term_structure
                                   )
        helper.setPricingEngine(engine)
        swaptions.append(helper)
    return swaptions    



def create_cap_helpers(data,index,term_structure,engine):
    # note that this is caps not caplets
    caps = []
    """
    CapHelper (const Period &length, const Handle< Quote > &volatility, const boost::shared_ptr< IborIndex > &index, 
    Frequency fixedLegFrequency, const DayCounter &fixedLegDayCounter, bool includeFirstSwaplet, 
    const Handle< YieldTermStructure > &termStructure, 
    CalibrationHelper::CalibrationErrorType errorType=CalibrationHelper::RelativePriceError)
    """
    fixed_leg_frequency = ql.Quarterly
    fixed_leg_dayCounter = ql.Actual360()
    includeFirstSwaplet = False
    for d in data:
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(d.volatility))
        helper = ql.CapHelper(ql.Period(d.length, ql.Years),
                              vol_handle,
                              index,
                              fixed_leg_frequency,
                              fixed_leg_dayCounter,
                              includeFirstSwaplet,
                              term_structure
                              )
        helper.setPricingEngine(engine)
        caps.append(helper)    

    return caps

##########################
###     Reporting
##########################

def calibration_report(helpers, data):
    columns = ["Model Price", "Market Price", "Implied Vol", "Market Vol", "Rel Error Price", "Rel Error Vols","Percentage Error"]
    report_data = []
    cum_err = 0.0
    cum_err2 = 0.0
    
    modelPrice = []
    blackPrice = []
    marketVol = []
    modelVol = []
    tenors = [x[0] for x in data]
    
    print tenors
    
    for i, s in enumerate(helpers):
        model_price = s.modelValue()
        market_vol = data[i].volatility
        black_price = s.blackPrice(market_vol)
        rel_error = model_price/black_price - 1.0
        implied_vol = s.impliedVolatility(model_price,
                                          1e-5, 50, 0.0, 0.50)
        rel_error2 = implied_vol/market_vol-1.0
        cum_err += rel_error*rel_error
        cum_err2 += rel_error2*rel_error2
        
        percentage_error = (implied_vol - market_vol)*100/market_vol
        
        modelPrice.append(model_price);
        blackPrice.append(black_price)
        modelVol.append(implied_vol)
        marketVol.append(market_vol)
        
        report_data.append((model_price, black_price, implied_vol,
                            market_vol, rel_error, rel_error2,percentage_error))
    print "Cumulative Error Price: %7.5f" % math.sqrt(cum_err)
    print "Cumulative Error Vols : %7.5f" % math.sqrt(cum_err2)
    
    plt.plot(tenors,marketVol,'-o',label="marketvol");
    plt.plot(tenors,modelVol,'-o',label="modelvol");
    plt.legend()
            
    return pd.DataFrame(report_data,columns= columns, index=['']*len(report_data))



##########################
###     General Settings
##########################

today = ql.Date(30, ql.December, 2016);
settlement= ql.Date(30, ql.December, 2016);
ql.Settings.instance().evaluationDate = today;
tsCurve = ql.YieldTermStructureHandle(ql.FlatForward(settlement,0.04875825,ql.Actual360()))
#index = ql.Euribor3M(term_structure)
#index = ql.USDLibor(ql.Period(3,ql.Months),term_structure)

calendar = ql.TARGET()
daycounter = ql.Actual360();
                        
###########################
###     Term Structure Construction
###########################

data = pd.read_csv(input_location + "ois_rates" + ".csv",header = None)

ts_dates = [ql.Date(x) for x in data.iloc[:,0].astype(int)]
ts_zerorates = data.iloc[:,1].tolist()
#tsCurve = ql.YieldTermStructureHandle(ql.ZeroCurve(ts_dates, ts_zerorates, daycounter, calendar))


index = ql.USDLibor(ql.Period(3,ql.Months),tsCurve)



########################################
###    MODEL 1 HW1F constant parameters
########################################

model = ql.HullWhite(tsCurve,0.05,0.0001)
engine = ql.AnalyticCapFloorEngine(model,tsCurve)           #CapEngine
#engine = ql.JamshidianSwaptionEngine(model,tsCurve)


#swaptions = create_swaption_helpers(swaption_data, index, tsCurve, engine)
caps= create_cap_helpers(cap_data,index,tsCurve,engine)

optimization_method = ql.LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)
model.calibrate(caps, optimization_method, end_criteria, ql.NoConstraint(), [], [True, False])
#model.calibrate(swaptions, optimization_method, end_criteria, ql.NoConstraint(), [], [True, False])


print "alpha: {}   sigma:  {}".format(model.params()[0],model.params()[1])

print calibration_report(caps,cap_data)
#print calibration_report(swaptions,swaption_data)

########################################
###    MODEL 1 HW1F GSR Model  (Python can only calibrate to Swaptions/ C++ can calibrate to Caps)
########################################

"""
a = ql.QuoteHandle(ql.SimpleQuote(0.01))
b = ql.QuoteHandle(ql.SimpleQuote(0.02))
c = ql.QuoteHandle(ql.SimpleQuote(0.03))
vol_quotes = [a,b,c]


date_vectors = [today+100,settlement+200]

mean_reversion_quote = [ql.QuoteHandle(ql.SimpleQuote(0.05))]

model2 = ql.Gsr(term_structure,date_vectors,vol_quotes,mean_reversion_quote,60)
engine = ql.AnalyticCapFloorEngine(model2)

"""





















