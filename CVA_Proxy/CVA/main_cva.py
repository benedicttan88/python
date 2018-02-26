# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:52:16 2017

@author: WB512563
"""

import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import method1; import method2; import method3;
import plotting

import pickle
import seaborn as sns

location = "D:\\python\\CVA_Proxy\\Input\\"
output_location = "D:\\python\\CVA_Proxy\\Output\\"

#filename = "20090615CDSComposite"
#filename2 = "cds_data_30Jun2016"
#
#df = pd.read_excel(location + filename + ".xlsx",parse_cols="B:T,AH:AL")
##df = pd.read_csv(location + filename2 + ".csv",skiprows=2)
#
#df = df[(df.Tier == "SNRFOR") & (df.Ccy == "USD") & (df.DocClause == "CR")]
#
#OTHER = df[(df.Sector == "Financials") | (df.Sector == "Government")]
#OTHER_CRVES = df[~df.index.isin(OTHER.index)]
#df.ix[OTHER_CRVES.index, "Sector"] = "Others"

      
# read python dict back from the file
pkl_file = open(location + "myfile.pkl", 'rb')
df = pickle.load(pkl_file)
pkl_file.close()

df = df[(df.Tier == "SNRFOR") & (df.Ccy == "USD") & (df.DocClause == "CR")]
df.dropna(subset=["ImpliedRating"],inplace=True)
     




##################################
######## Compute Statistics

##Order by Implied Rating
order =["AA","A","BBB","BB","B","CCC"]
count_of_CDS_DATA = df.loc[:,"Spread6m":"ImpliedRating"].groupby("ImpliedRating").count()
count_of_CDS_DATA = count_of_CDS_DATA.reindex(order)

#Order by Sector | Region | ImpliedRating
a = df.loc[:,"Ticker":"ImpliedRating"].groupby(["Sector","Region","ImpliedRating"]).count()
a.to_excel(output_location + "Sector Classification.xlsx")
#need to reorder


#
reload(plotting)
#plotting.boxplot(df,order)
#plotting.boxplot_Sector(df)
#plotting.boxplot_Region(df)
plotting.barplot(count_of_CDS_DATA)


################################


###################################
#### CHOOSE COUNTRY
###################################

Ticker = "BRAZIL"
kfold = True


#######################################
#### Method 1 
# Leave all Blanks out of the equation
#######################################
#reload(method1)
#dataframe_x , cont , names1 = method1.cva_proxy(df,Ticker,kfold)
#
#ax = cont.plot(title="Method1")
#ax.lines[1].set_linewidth(4)

                           
#######################################
#### Method 2
# Leave all Blanks out of the equation
#######################################
#reload(method2)
#dataframe_x2 , cont2, names2 = method2.cva_proxy(df,Ticker,kfold)
#
#ax2 = cont2.plot(title="Method2")
#ax2.lines[1].set_linewidth(4)


#####################################
#### MEthod 3
# Spread from 5y Tenor
#####################################

#reload(method3)
#
#method3.cva_proxy(df,Ticker,kfold)




    





    