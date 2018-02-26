# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:07:20 2017

@author: WB512563
"""

import pandas as pd


def to_excel(schedule_list,location):
    
    """
    Printing to excel schedule
    """
    
    
    excel_dataframe_schedules = pd.DataFrame()
    for i in range(len(schedule_list)):
        a = [d.serialNumber() for i,d in enumerate(schedule_list[i])]
        excel_dataframe_schedules = excel_dataframe_schedules.append(pd.DataFrame(a).T, ignore_index = True)
    
    excel_dataframe_schedules = excel_dataframe_schedules.T    
    #excel_dataframe_schedules.columns = [d.serialNumber() for d in start_dates]
    excel_dataframe_schedules.to_excel(location,'Coupon_Schedules',index=False)
    
    
    
    