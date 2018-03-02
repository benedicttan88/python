# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:08:39 2017

@author: WB512563
"""

import QuantLib as ql
import numpy as np
import pandas as pd

#import Instrument.swap as SWAP
from Instrument.swap import QL_makeVanillaSwap


today = ql.Date(15,ql.July,2017)

####  Read in Discounting Curve


rate = ql.SimpleQuote(0.03)
rate_handle = ql.QuoteHandle(rate)
dc = ql.Actual365Fixed()
yts = ql.FlatForward(today, rate_handle, dc)
yts.enableExtrapolation()
hyts = ql.RelinkableYieldTermStructureHandle(yts)
t0_curve = ql.YieldTermStructureHandle(yts)
euribor6m = ql.Euribor6M(hyts)

####    Create Swap Instrument
Notional = 10000000
fixedRate = 0.0234
floatSpread = 0.005
indexCurve = ql.Euribor6M(hyts)
flavour = ql.VanillaSwap.Payer

##  Fixed Schedule
PAY_LEG_CALENDAR = ql.TARGET()
PAY_LEG_FREQ = ql.Semiannual
PAY_LEG_EFFECTIVE_DATE = ql.Date(15,ql.July,2017)
PAY_LEG_MATURITY = ql.Date(15,ql.July,2040)
PAY_LEG_BUS_CONV = ql.ModifiedFollowing

fixedSchedule = ql.Schedule(PAY_LEG_EFFECTIVE_DATE, PAY_LEG_MATURITY,
                       ql.Period(PAY_LEG_FREQ),
                        PAY_LEG_CALENDAR,
                        PAY_LEG_BUS_CONV,
                        ql.Unadjusted,
                        ql.DateGeneration.Forward,
                        False);

##  Float Schedule
RECEIVE_LEG_CALENDAR = ql.TARGET()
RECEIVE_LEG_FREQ = ql.Semiannual
RECEIVE_LEG_EFFECTIVE_DATE = ql.Date(15,ql.July,2017)
RECEIVE_LEG_MATURITY = ql.Date(15,ql.July,2040)
RECEIVE_LEG_BUS_CONV = ql.ModifiedFollowing

floatSchedule = ql.Schedule(RECEIVE_LEG_EFFECTIVE_DATE, RECEIVE_LEG_MATURITY,
                       ql.Period(RECEIVE_LEG_FREQ),
                        RECEIVE_LEG_CALENDAR,
                        RECEIVE_LEG_BUS_CONV,
                        ql.Unadjusted,
                        ql.DateGeneration.Forward,
                        False);
                           

swap_instrument = QL_makeVanillaSwap(flavour, Notional, fixedRate, fixedSchedule, floatSchedule, euribor6m, floatSpread)



####    Create Model


