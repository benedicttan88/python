# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:59:41 2017

@author: WB512563
"""

import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


location = "D:\\python\\CVA_Proxy\\Input\\"

filename = "cds_data_30Jun2016"

df = pd.read_csv(location + filename + ".csv",skiprows=2)
df = df[(df.Tier == "SNRFOR") & (df.DocClause == "CR14")]

OTHER = df[(df.Sector == "Financials") | (df.Sector == "Government")]
OTHER_CRVES = df[~df.index.isin(OTHER.index)]
df.ix[OTHER_CRVES.index, "Sector"] = "Others"
     
     
df.drop("AvRating",axis=1,inplace= True)
df.dropna(inplace = True)


#df = df[~(df.RedCode == "YC58FN")].copy()

#Spread1y = np.log(df.Spread6m.values)
#
##A = df[["Rating6m","Region","Country","Sector"]]
#A = df[["Rating6m","Region","ImpliedRating"]]
#
#A_encode = pd.get_dummies(A)
#pp = A_encode
#columns = A_encode.columns
#A_encode = A_encode.values
#
#model = LinearRegression()
#model.fit(A_encode, Spread1y)
#x = model.coef_
#difference = Spread1y - np.dot(A_encode,x)
#d1 = np.linalg.norm(difference)
#
#
#x1 = np.linalg.lstsq(A_encode,Spread1y)[0]
#difference2 = Spread1y - np.dot(A_encode,x1)
#d2 = np.linalg.norm(difference2)
#
#
#
#a = np.zeros(len(columns))
#index = [1,9,23]
#for i in range(len(index)):
#    a[index] = 1
#     
#     
#estimated_spread = exp(np.dot(a,x1))
#"Spread1y","Spread2y",



###################################
#### CHOOSE COUNTRY
###################################

Ticker = "VENZ"
c = df[df.Ticker == Ticker]
original_spreads = c.loc[:,"Spread6m":"Spread30y"].values.tolist()[0]
#Tenor_Rating = c["Rating" + "2y"].values[0]
Sector = "Sector_" + c["Sector"].values[0]
Region =  "Region_" + c["Region"].values[0]
Implied_Rating =  "ImpliedRating_" + c["ImpliedRating"].values[0]




#######################################
##### CHOOSE SIMILAR COUNTRIES TO THIS
#######################################

myset = df[(df.Region == c["Region"].values[0]) & (df.ImpliedRating == c["ImpliedRating"].values[0])].head()
myset_names = myset.Ticker.values
myset_data = myset.loc[:,"Spread6m":"Spread30y"].T
myset_data.columns = myset_names


## total number of tables to make: 11
# 8 to 19
columnnames = df.columns[8:19]
Tenors = columnnames.str[6:].tolist()


dataframe_x = pd.DataFrame()
container = []          #craete container to store items

for i in np.arange(11):
    print i
    print "Tenor: {}".format(Tenors[i])
    #set b
    print "b: " , columnnames[i]
    b = np.log(df[columnnames[i]].values)
    #set A
    Rating = "Rating" 
    #A = df[["Sector","Region","ImpliedRating"]]
    A = df[["Region","ImpliedRating"]]
    A_encode = pd.get_dummies(A)
    columns = A_encode.columns.tolist()
    A_encode = A_encode.values
    
    A_encode = np.column_stack( (A_encode,np.ones(A_encode.shape[0]) ))
    
    
    x1 = np.linalg.lstsq(A_encode,b)[0]
    difference2 = b - np.dot(A_encode,x1)
    d2 = np.linalg.norm(difference2)
    
    print "||difference|| : ", d2
    
    
    
    Tenor_Rating = "Rating" + Tenors[i] + "_" + c["Rating" + Tenors[i]].values[0]
    
    
    
    a = np.zeros(len(columns)+1)
    
    #index_rating = columns.index(Tenor_Rating)
    #index_sector = columns.index(Sector)
    index_region = columns.index(Region)
    index_ImpliedRating = columns.index(Implied_Rating)
    index_Global = len(columns)
    index = [index_region,index_ImpliedRating,index_Global]
    for i in range(len(index)):
        a[index] = 1
    estimated_spreadz = np.exp(np.dot(a,x1))
    
    
    dataframe_x = dataframe_x.append(pd.DataFrame(np.exp(x1)).T)
    container.append(estimated_spreadz)
    
    
    




cont = pd.DataFrame([original_spreads,container]).T
cont.columns = ["Original","LstSq"]
cont.index = columnnames

cont = pd.concat([cont,myset_data],axis=1)




columns.append("Global")
dataframe_x = dataframe_x.T
dataframe_x.index = columns
dataframe_x.columns = columnnames



plt.figure()
ax = cont.plot()
ax.lines[1].set_linewidth(4)







    