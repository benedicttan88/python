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
pkl_file = open(location + "myfile.pkl", 'rb')
df = pickle.load(pkl_file)
pkl_file.close()

#ticker_list = pd.read_csv(location + "ticker_list_to_bootstrap" + ".csv",skiprows=0,header=None)
#ticker_list = ticker_list[0].tolist()

###################################
# Tickername
##################################
TickerName = "BRAZIL"

#---------------------------------------------------


#drop all the Blanks out of the eequation
#df.dropna(inplace = True)

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

##########################################################
######################  convert tenors to months ---> list
##########################################################
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
        
        
## total number of tables to make: 11
# 8 to 19
columnnames = df.columns[7:18]
Tenors = columnnames.str[6:].tolist()

storing_b = []

new_outer = []
CV_score_list = []
dataframe_x = pd.DataFrame()
container = []          #craete container to store items

for i in np.arange(11):
    
    
    df.dropna(subset=["Spread"+Tenors[i]],inplace=True)
    df.dropna(subset=["ImpliedRating"],inplace=True)
    
    
#    ###### Switch on standard deviation for every ImpliedRating #######################################
#    listofstd = df.groupby("ImpliedRating").std().iloc[:,i].to_dict()
#    listofmean = df.groupby("ImpliedRating").mean().iloc[:,i].to_dict()
#    totalListofIndex = []
#    for x,y in listofmean.items():
#        #print x
#        #print "lower: {} , mean: {} , upper: {}".format(y - 5*listofstd[x] ,y ,y + 5*listofstd[x])
#        lower_std = y - 5*listofstd[x]
#        upper_std = y + 5*listofstd[x]
#        listofindex = df[(df.ImpliedRating == x) & ( lower_std <= df[columnnames[i]] ) & ( df[columnnames[i]] <= upper_std)].index
#        totalListofIndex.extend(listofindex.tolist())
#    df = df.ix[totalListofIndex]
#    ###################################################################################################
    
    print i
    print "Tenor: {}".format(Tenors[i])
    #set b
    print "b: " , columnnames[i]
    b = np.log(df[columnnames[i]].values)
    
    storing_b.append(len(b))
    #b = df[columnnames[i]].values
    #set A
    A = df[["Sector","Region","ImpliedRating"]]
    #A = df[["Region","ImpliedRating"]]
    encoded_A = pd.get_dummies(A)
    columns = encoded_A.columns.tolist()
    
    A_encode = encoded_A.values
    A_encode = np.column_stack( (A_encode,np.ones(A_encode.shape[0]) ))             #Add the Global ones
    

    #######################################
    
    ## 2 norm
    reg = sklearn.linear_model.Ridge(alpha=2.0,fit_intercept=False)
    reg.fit(A_encode,b)
    x1 = reg.coef_
#
#    reg = sklearn.svm.LinearSVR(fit_intercept=False,C=1.0)
#    reg.fit(A_encode,b)
#    x1 = reg.coef_
    
#    reg = sklearn.linear_model.BayesianRidge(fit_intercept=False)
#    reg.fit(A_encode,b)
#    x1 = reg.coef_
    
#    reg = sklearn.linear_model.Lasso(fit_intercept=False,alpha=0.01)
#    reg.fit(A_encode,b)
#    x1 = reg.coef_
    
#    reg = sklearn.linear_model.ElasticNet(alpha=2.0,l1_ratio=0.0,fit_intercept=False)
#    reg.fit(A_encode,b)
#    x1 = reg.coef_
    
#    reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
#    reg.fit(A_encode,b)
#    x1 = reg.coef_


    #######################################
    
    #x1 = np.linalg.lstsq(A_encode,b)[0]
#        
#        diff =  x1 - x1_scikit
#        print "diff: {}".format(diff)
    
    #print "lstsq: {} , scikit: {}".format(x1,x1_scikit)

    ####################################    
    ## ----- CV AREA
#    scores = []
#    kf = KFold(n_splits=10)
#    for train_index, test_index in kf.split(A_encode,b):
#        reg.fit(A_encode[train_index],b[train_index])
#        score = reg.score(A_encode[test_index],b[test_index])
#        print "score: {}".format(score)
#        scores.append(score)
#    print "Average Score: {}".format(np.mean(scores))
#    CV_score_list.append(np.mean(scores))
    ####################################
    
    ####################################
    # MY KFOLD
#    nTimes = 1
#    nGroups = 5
#    scores = []
#    for i in np.arange(nTimes):
#        #scores = []
#        kf = KFold(n_splits=nGroups,shuffle=True,random_state=3)
#        for train_index, test_index in kf.split(A_encode,b):
#            reg.fit(A_encode[train_index],b[train_index])
#            x1_reg = reg.coef_
#            
#            #difference = b - Ax
#            diff = np.exp(b[test_index]) - np.exp(reg.predict(A_encode[test_index]))
#            score = np.mean(np.square(diff))
#            #print "score: {}".format(score)
#        
#        scores.append(score)
#        
#    #print "Average Score: {}".format(np.mean(scores))
#    CV_score_list.append(np.sum(scores))
    
    nGroups = 5
    scores = []
    kf = KFold(n_splits=nGroups,shuffle=True,random_state=3)
    for train_index, test_index in kf.split(A_encode,b):
        reg.fit(A_encode[train_index],b[train_index])
        x1_reg = reg.coef_
        
        #difference = b - Ax
        diff = np.exp(reg.predict(A_encode[test_index]))
        #diff = np.exp(b[test_index]) - np.exp(reg.predict(A_encode[test_index]))
        score = np.mean(np.square(diff))

        scores.append(score)
        
    #CV_score_list.append(np.sum(scores))
    CV_score_list.extend(scores)
    
    a = np.zeros(len(columns)+1)
    
    index_sector = columns.index(Sector)
    index_region = columns.index(Region)
    index_ImpliedRating = columns.index(Implied_Rating)
    index_Global = len(columns)
    index = [index_sector, index_region, index_ImpliedRating, index_Global]
    for i in range(len(index)):
        a[index] = 1
    estimated_spreadz = np.exp(np.dot(a,x1))
    print "spread: {}".format(estimated_spreadz)
    #estimated_spreadz = np.dot(a,x1)
    
    dataframe_x = dataframe_x.append(pd.DataFrame(x1).T)
    #dataframe_x = dataframe_x.append(pd.DataFrame(np.exp(x1)).T)
    container.append(estimated_spreadz)
    

print "Final CV Score(SUM): {}".format(np.sum(CV_score_list))
print "Final CV Score(MEAN): {}".format(np.mean(CV_score_list))


################### Nelson Siegel Fitting Area ################################
#lambda_ = 0.0609
#
#A = zip(time_to_maturity,container)
#for i, (x,y) in enumerate(A):
#    #print x
#    A[i] =  A[i] + (1,short(lambda_,x),medium(lambda_,x))
#dd = pd.DataFrame(A)
#dd.dropna(inplace = True)
#b = dd.iloc[:,1].values
#W = dd.iloc[:,2:].values
#           
#reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
#reg.fit(W,b)
#NS_coef = reg.coef_
#
#
#NS_Curve = [NS(NS_coef[0],NS_coef[1],NS_coef[2],lambda_,o) for o in time_to_maturity]
#################################################################################



cont = pd.DataFrame([original_spreads,container]).T
cont.columns = ["Original","Model2"]
cont.index = columnnames

cont = pd.concat([cont,myset_data],axis=1)


columns.append("Global")
dataframe_x = dataframe_x.T
dataframe_x.index = columns
dataframe_x.columns = columnnames


ax = cont.plot(title="CDS Spreads of Similar Sector, Region, ImpliedRating\n Sector:{} Region: {} , Rating: {}".format(c["Sector"].values[0],c["Region"].values[0],c["ImpliedRating"].values[0]))
ax.lines[1].set_linewidth(5)
ax.set_ylabel("CDS Spread")
ax.set_xlabel("Tenor")


#
#NS_Curve_Final = [NS(estimated_b1,estimated_b2,estimated_b3,lambda_,o) for o in time_to_maturity]
#cont = pd.DataFrame([original_spreads,container,NS_Curve,NS_Curve_Final]).T
#cont.columns = ["Original","(Approach 1) Model2","(Approach 1) Model2 + NS","Approach 2"]
#cont.index = columnnames
#cont = pd.concat([cont,myset_data],axis=1)
