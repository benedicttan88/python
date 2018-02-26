# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 23:34:56 2017

@author: Benedict Tan
"""

##With KFold Cross Validation


import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import method1
import method2
import plotting

import sklearn.svm
import sklearn.neural_network
#from sklearn.svm import SVR


import pickle

from sklearn.model_selection import KFold


def short(lmbda,tau):
    temp = lmbda*tau
    return (1 - np.exp(-temp))/temp

def medium(lmbda,tau):
    temp = lmbda*tau
    return ((1 - np.exp(-temp))/temp) - np.exp(-temp) 


def NS(b1,b2,b3,lmbda,tau):
    
    return b1 + b2*short(lmbda,tau) + b3*medium(lmbda,tau)


location = "D:\\python\\CVA_Proxy\\Input\\"
pkl_file = open(location + "myfile3.pkl", 'rb')
df = pickle.load(pkl_file)
pkl_file.close()
df = df.reset_index(drop=True)


#count index to drop
list_to_drop = []
for index, row in df.iterrows():
    if row["Spread6m":"Spread30y"].count() <= 5:
        list_to_drop.append(index)

df.drop(df.index[list_to_drop],inplace=True)



###################################
# Tickername
##################################
TickerName = "JPM"


###################################
#### CHOOSE COUNTRY
###################################
c = df[df.Ticker == TickerName]
print c
original_spreads = c.loc[:,"Spread6m":"Spread30y"].values.tolist()[0]

Sector = "Sector_" + c["Sector"].values[0]
Region =  "Region_" + c["Region"].values[0]
Implied_Rating =  "ImpliedRating_" + c["ImpliedRating"].values[0]



#######################################
##### CHOOSE SIMILAR COUNTRIES TO THIS
#######################################

print "choosing areas similar to: {} {} {}".format(c["Sector"].values[0],c["Region"].values[0],c["ImpliedRating"].values[0])

myset = df[(df.Sector == c["Sector"].values[0]) & (df.Region == c["Region"].values[0]) & (df.ImpliedRating == c["ImpliedRating"].values[0])]
myset_names = myset.Ticker.values
print myset_names
myset_data = myset.loc[:,"Spread6m":"Spread30y"].T
myset_data.columns = myset_names






#convert tenors to months ---> list
termstructurelist = df.columns[7:18].str.upper().str[6:].tolist()

months_to_maturity = []
time_to_maturity = []
tenors = [];
for i in range(len(termstructurelist)):
    if (termstructurelist[i][-1] == 'M'):
        n = int(termstructurelist[i][:-1])
        months_to_maturity.append(n)
        time_to_maturity.append(float(n)/12.0)
    elif (termstructurelist[i][-1] == 'Y'):
        
        n = int(termstructurelist[i][:-1])
        months_to_maturity.append(n*12)
        time_to_maturity.append(float(n))
        
#start solving for b1,b2,b3 one by one and store them as vector
lambda_ = 0.0609

listof_b1 = []
listof_b2 = []
listof_b3 = []

###############  Choose 1 from each sector region and rating

Sector_List = ["Government"]
Region_List = ["N.Amer","Europe","Africa","Asia","Oceania","OffShore","E.Europe","Lat.America","MiddleEast","Caribbean","India","Supra"]
Rating_List= ["AA","A","BBB","BB","B","CCC"]

for w in Sector_List:
    for j in Region_List:
        for k in Rating_List:
            
            print "Sector: {} , Region: {} , Rating: {}".format(w,j,k)
            
            D = df[(df.Sector == w) & (df.Region == j) & (df.ImpliedRating == k)]
            if D.empty:
                print " "
            else:
                print "Here"
                D = D.sample(1)
                Ticker = D["Ticker"].values[0]
                print "Ticker: {}".format(Ticker)
                A = zip(time_to_maturity,D.loc[:,"Spread6m":"Spread30y"].values[0])
                for i, (x,y) in enumerate(A):
                    #print x
                    A[i] =  A[i] + (1,short(lambda_,x),medium(lambda_,x))
                dd = pd.DataFrame(A)
                dd.dropna(inplace = True)
                b = dd.iloc[:,1].values
                W = dd.iloc[:,2:].values
                
                reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
                reg.fit(W,b)
                x = reg.coef_
                
                #Plotting now
                plt.figure()
                last_maturity = time_to_maturity[-1]        
                t = np.linspace(0,last_maturity,last_maturity+1)
                NS_Curve = [NS(x[0],x[1],x[2],lambda_,o) for o in t]
                plt.plot(t,NS_Curve,label="NS Fitted Curve")        
                
                original = D.loc[:,"Spread6m":"Spread30y"].T
                plt.plot(time_to_maturity,original,label="Original")
                plt.legend()
                plt.title("Sector: {}, Region: {}, Rating: {}\n Ticker: {}".format(w,j,k,Ticker))
                


for index,row in df.iterrows():
    if row["Ticker"] == "JPM":
        A = zip(time_to_maturity,row["Spread6m":"Spread30y"].tolist())
        for i, (x,y) in enumerate(A):
            print x
            A[i] =  A[i] + (1,short(lambda_,x),medium(lambda_,x))
        dd = pd.DataFrame(A)
        dd.dropna(inplace = True)
        b = dd.iloc[:,1].values
        W = dd.iloc[:,2:].values
        
        reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
        reg.fit(W,b)
        JPM_x = reg.coef_
        

plt.figure()
last_maturity = time_to_maturity[-1]        
t = np.linspace(0,last_maturity,last_maturity+1)
NS_Curve = [NS(JPM_x[0],JPM_x[1],JPM_x[2],lambda_,o) for o in t]
plt.plot(t,NS_Curve,label="NS Fitted Curve")        

original = c.loc[:,"Spread6m":"Spread30y"].T
plt.plot(time_to_maturity,original,label="Original")
plt.legend()
plt.title("Sector: {}, Region: {}, Rating: {}".format("Financials","Europe","AA"))
