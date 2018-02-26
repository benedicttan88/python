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

def Loss(x):
    return np.square(x[0] - x[1])


location = "D:\\python\\CVA_Proxy\\Input\\"
#pkl_file = open(location + "myfile3.pkl", 'rb')
#df = pickle.load(pkl_file)
#pkl_file.close()
df = pd.read_pickle(location + "myfile3.pkl")
df = df.reset_index(drop=True)

#df = pd.from_pickle(location + "myfile3.pkl",'rb')

#count index to drop
list_to_drop = []
for index, row in df.iterrows():
    if row["Spread6m":"Spread30y"].count() <= 5:
        list_to_drop.append(index)

df.drop(df.index[list_to_drop],inplace=True)

df = df.reset_index(drop=True)

###################################
# Tickername
##################################
TickerName = "BRAZIL"


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

for index,row in df.iterrows():
    A = zip(time_to_maturity,row["Spread6m":"Spread30y"].tolist())
    for i, (x,y) in enumerate(A):
        A[i] =  A[i] + (1,short(lambda_,x),medium(lambda_,x))
    dd = pd.DataFrame(A)
    dd.dropna(inplace = True)
    b = dd.iloc[:,1].values
    W = dd.iloc[:,2:].values
    
    reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    #reg = sklearn.linear_model.Ridge(alpha=3.0,fit_intercept=False)
    reg.fit(W,b)
    x = reg.coef_
    listof_b1.append(x[0])
    listof_b2.append(x[1])
    listof_b3.append(x[2])
    if row["Ticker"] == TickerName:
        TickerX = x

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

        


#now Get your A_encode and solve for each x_hat that characterises the level,slope,curvature
        

A = df[["Ticker","Sector","Region","ImpliedRating"]]
encoded_A = pd.get_dummies(A)
columns = encoded_A.columns.tolist()


A_encode = encoded_A.values
reg1 = sklearn.linear_model.LinearRegression(fit_intercept=False)
#reg1 = sklearn.linear_model.Ridge(alpha=2.0,fit_intercept=False)
reg1.fit(A_encode,listof_b1)
level_X = reg1.coef_

reg2 = sklearn.linear_model.LinearRegression(fit_intercept=False)
#reg2 = sklearn.linear_model.Ridge(alpha=2.0,fit_intercept=False)
reg2.fit(A_encode,listof_b2)
slope_X = reg2.coef_

reg3 = sklearn.linear_model.LinearRegression(fit_intercept=False)
#reg3 = sklearn.linear_model.Ridge(alpha=2.0,fit_intercept=False)
reg3.fit(A_encode,listof_b3)
curvature_X = reg3.coef_

###################################################
######################### MY KFOLD
####################################################
#nTimes = 1
#nGroups = 20
#scores = []
#
#listof_b1 = np.array(listof_b1)
#listof_b2 = np.array(listof_b2)
#listof_b3 = np.array(listof_b3)
#
#Outer_CV_List = []
#kf = KFold(n_splits=nGroups,shuffle=True,random_state=3)
#for train_index, test_index in kf.split(A,listof_b1):
#    reg1.fit(A_encode[train_index],listof_b1[train_index])
#    x1_reg = reg1.coef_
#    reg2.fit(A_encode[train_index],listof_b2[train_index])
#    x2_reg = reg2.coef_    
#    reg3.fit(A_encode[train_index],listof_b3[train_index])
#    x3_reg = reg3.coef_    
#    
#    model_predicted_A = np.dot(A_encode[test_index],x1_reg)
#    model_predicted_B = np.dot(A_encode[test_index],x2_reg)
#    model_predicted_C = np.dot(A_encode[test_index],x3_reg)
#    
#    TickerList = A["Ticker"][test_index].tolist()
#    
#    #Need to get the Model Predicted Spreads (NS) at each tenors
#    CV_List_test_index_I_ = []
#    for i in range(len(model_predicted_A)):
#        #print i
#        Model_Predicted_Spreads = pd.DataFrame([NS(model_predicted_A[i],model_predicted_B[i],model_predicted_C[i],lambda_,o) for o in time_to_maturity])                               
#        Market_Spreads = df[df.Ticker == TickerList[i]].loc[:,"Spread6m":"Spread30y"].T
#        Market_Spreads.reset_index(drop=True, inplace=True)
#        temp_df = pd.concat([Model_Predicted_Spreads,Market_Spreads],axis=1,ignore_index=True)
#        temp_df.dropna(inplace=True)
#
#        temp_df["CV"] = temp_df.apply(Loss,axis=1)
#        CV_List_test_index_I_.append(temp_df["CV"].mean())
#        
#    Outer_CV_List.append(np.sum(CV_List_test_index_I_))
#
#print "Final CV Error: {}".format(np.mean(Outer_CV_List))




a = np.zeros(len(columns))

index_sector = columns.index(Sector)
index_region = columns.index(Region)
index_ImpliedRating = columns.index(Implied_Rating)

index = [index_sector, index_region, index_ImpliedRating]
for i in range(len(index)):
    a[index] = 1
estimated_b1 = np.dot(a,level_X)
estimated_b2 = np.dot(a,slope_X)
estimated_b3 = np.dot(a,curvature_X)

plt.figure()
last_maturity = time_to_maturity[-1]
t = np.linspace(0,last_maturity,last_maturity+1)

NS_Curve_Final = [NS(estimated_b1,estimated_b2,estimated_b3,lambda_,o) for o in t]
plt.plot(t,NS_Curve_Final,label="NS Fitted Curve Final")
NS_Curve_Start = [NS(TickerX[0],TickerX[1],TickerX[2],lambda_,o) for o in t]
plt.plot(t,NS_Curve_Start,label="NS Fitted Curve Start")


original = c.loc[:,"Spread6m":"Spread30y"].T


Similars_plot = pd.concat([original,myset_data],axis=1)

plt.plot(time_to_maturity,original,label="Original Spread")
plt.legend()



#### New Plot
myset_data.index = time_to_maturity
NS_Curve_Finals = [NS(estimated_b1,estimated_b2,estimated_b3,lambda_,o) for o in time_to_maturity]
NS_Curve_Finals = pd.DataFrame(NS_Curve_Finals,columns=["Approach 2"])
NS_Curve_Finals.index = time_to_maturity
myset_data = pd.concat([NS_Curve_Finals,myset_data],axis=1)
ax = myset_data.plot(title="CDS Spreads of Similar Sector, Region, ImpliedRating\n Sector:{} Region: {} , Rating: {}".format(c["Sector"].values[0],c["Region"].values[0],c["ImpliedRating"].values[0]))
ax.lines[0].set_linewidth(5)
ax.set_ylabel("CDS Spread")
ax.set_xlabel("Tenor")



fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
ax1.plot(t,NS_Curve_Final,label="NS Fitted Curve Final")
ax1.plot(t,NS_Curve_Start,label="NS Fitted Curve Intermediate")
ax1.plot(time_to_maturity,original,label="Original Spread")
ax1.set_ylabel("CDS Spread")
ax1.set_xlabel("Tenor")
ax1.legend()    
myset_data.plot(ax=ax2)      
ax2.lines[0].set_linewidth(5)
ax2.set_xlabel("Tenor")
title_ = "CDS Spreads of Similar Sector, Region, ImpliedRating\n Sector:{} Region: {} , Rating: {}".format(c["Sector"].values[0],c["Region"].values[0],c["ImpliedRating"].values[0])
fig.suptitle(title_, fontsize=14)
