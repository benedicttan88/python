# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:53:10 2017

@author: WB512563
"""

import numpy as np
import pandas as pd

import cds_ts
import QuantLib as ql
import matplotlib.pyplot as plt


### TESTING DIFFERENT MODELS

## HW1F Constant
## GSR
## CIR
## Heston


############################### General Settings ###########################################

location = "D:\\worldbank\\hazard_rates_bootstrap\\input\\"
ois_location = "D:\\python\\Input\\Yieldcurve\\"
index_curve_location = "D:\\python\\Input\\Yieldcurve\\"

ois_file = "ois_df_30jun2017"
index_curve_file = "usd_lib6m_df_30jun2017"


############################  Discounting OIS Curve ########################################

ts_dates = cds_ts.get_dates(ois_location,ois_file);
ts_rates = cds_ts.get_rates(ois_location,ois_file);
                           
#tsCurve = ql.YieldTermStructureHandle(ql.ZeroCurve(ts_dates, ts_rates, DayCounter, calendar))
tsCurve = ql.YieldTermStructureHandle(ql.DiscountCurve(ts_dates, ts_rates, ql.Actual360(), ql.TARGET()))



################  HW1F Constant ###############

mean_reversion = 0.05
sigma = 0.02

HW1F_Constant_Process = ql.HullWhiteProcess( tsCurve, mean_reversion , sigma )


###############    GSR Process ################

times = [ 10 ]
vols = [ 0.02 , 0.05  ]
reversions = [ 0.05 ]

T = 30

GSR_Process = ql.GsrProcess(times , vols , reversions , T)

############# Simulation Paths ##############################################################

seed = 22
generator = ql.MersenneTwisterUniformRng(seed)

timestep = 360;
length = 30     #years



rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
seq = ql.GaussianPathGenerator(GSR_Process, length, timestep, rng, False)



def generate_paths(num_paths, timestep):
    
    arr = np.zeros((num_paths, timestep+1))
    for i in range(num_paths):
        sample_path = seq.next()
        path = sample_path.value()
        time = [path.time(j) for j in range(len(path))]
        value = [path[j] for j in range(len(path))]
        arr[i, :] = np.array(value)
    return np.array(time), arr



num_paths = 10
time, paths = generate_paths(num_paths, timestep)
for i in range(num_paths):
    plt.plot(time, paths[i, :], lw=0.8, alpha=0.6)
plt.title("Hull-White Short Rate Simulation")
plt.show()

























