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
original_df = pickle.load(pkl_file)
pkl_file.close()
original_df.dropna(subset=["ImpliedRating"],inplace=True)
original_df.dropna(subset=["Recovery"],inplace=True)
original_df = original_df.reset_index(drop=True)

#filename = "20090615CDSComposite"
#df = pd.read_excel(location + filename + ".xlsx",parse_cols="B:T,AH:AL")

temp_df = pd.read_csv(location + "ticker_list_to_bootstrap_Financials.csv",names=["Ticker"])
ticker_list = temp_df.Ticker.tolist()


#count index to drop
#list_to_drop = []
#for index, row in original_df.iterrows():
#    if (row["Spread6m":"Spread30y"].count() < 5) :
#        list_to_drop.append(index)


#temp_df = original_df.ix[original_df.index[list_to_drop]]
#ticker_list = temp_df[temp_df.Sector == "Government"].Ticker.tolist()
#ticker_list.remove("FIJI")

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
#c = original_df[original_df.Ticker == TickerName]
#print c
#original_spreads = c.loc[:,"Spread6m":"Spread30y"].values.tolist()[0]
#
#Sector = "Sector_" + c["Sector"].values[0]
#Region =  "Region_" + c["Region"].values[0]
#Implied_Rating =  "ImpliedRating_" + c["ImpliedRating"].values[0]


#######################################
##### CHOOSE SIMILAR COUNTRIES TO THIS
#######################################

#print "choosing areas similar to: {} {} {}".format(c["Sector"].values[0],c["Region"].values[0],c["ImpliedRating"].values[0])
#
#myset = original_df[(original_df.Sector == c["Sector"].values[0]) & (original_df.Region == c["Region"].values[0]) & (original_df.ImpliedRating == c["ImpliedRating"].values[0])]
#myset_names = myset.Ticker.values
#print myset_names
#myset_data = myset.loc[:,"Spread6m":"Spread30y"].T
#myset_data.columns = myset_names


##########################################################
######################  convert tenors to months ---> list
##########################################################
termstructurelist = original_df.columns[7:18].str.upper().str[6:].tolist()

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




spread_list = pd.DataFrame()
spread_list_NS = pd.DataFrame()

for tickerName in ticker_list:
    
    df = original_df.copy()
    
    print "ticker: {}".format(tickerName)
    Sector = "Sector_" + df[df.Ticker == tickerName]["Sector"].values[0]
    Region = "Region_" + df[df.Ticker == tickerName]["Region"].values[0]
    Implied_Rating =  "ImpliedRating_" + df[df.Ticker == tickerName]["ImpliedRating"].values[0]
    
    
    ## total number of tables to make: 11
    # 8 to 19
    columnnames = df.columns[7:18]
    Tenors = columnnames.str[6:].tolist()
    
    CV_score_list = []
    dataframe_x = pd.DataFrame()
    container = []          #craete container to store items
    container_NS = []
    
    for i in np.arange(11):
        
        df.dropna(subset=["Spread"+Tenors[i]],inplace=True)

        
        ###### Switch on standard deviation for every ImpliedRating #######################################
#        listofstd = df.groupby("ImpliedRating").std().iloc[:,i].to_dict()
#        listofmean = df.groupby("ImpliedRating").mean().iloc[:,i].to_dict()
#        totalListofIndex = []
#        for x,y in listofmean.items():
#            #print x
#            #print "lower: {} , mean: {} , upper: {}".format(y - 5*listofstd[x] ,y ,y + 5*listofstd[x])
#            lower_std = y - 5*listofstd[x]
#            upper_std = y + 5*listofstd[x]
#            listofindex = df[(df.ImpliedRating == x) & ( lower_std <= df[columnnames[i]] ) & ( df[columnnames[i]] <= upper_std)].index
#            totalListofIndex.extend(listofindex.tolist())
#        df = df.ix[totalListofIndex]
        ###################################################################################################
        
        #print i
        #print "Tenor: {}".format(Tenors[i])
        #set b
        #print "b: " , columnnames[i]
        b = np.log(df[columnnames[i]].values)
        #set A
        A = df[["Sector","Region","ImpliedRating"]]
        encoded_A = pd.get_dummies(A)
        columns = encoded_A.columns.tolist()
        
        A_encode = encoded_A.values
        A_encode = np.column_stack( (A_encode,np.ones(A_encode.shape[0]) ))             #Add the Global ones
        
    
        #######################################
        
        ### 2 norm
        reg = sklearn.linear_model.Ridge(alpha=2.0,fit_intercept=False)
        reg.fit(A_encode,b)
        x1 = reg.coef_
    
#        reg = sklearn.svm.LinearSVR(fit_intercept=False,C=1.0)
#        reg.fit(A_encode,b)
#        x1 = reg.coef_
        
    #    reg = sklearn.linear_model.BayesianRidge(fit_intercept=False)
    #    reg.fit(A_encode,b)
    #    x1 = reg.coef_
        
    #    reg = sklearn.linear_model.Lasso(fit_intercept=False,alpha=0.01)
    #    reg.fit(A_encode,b)
    #    x1 = reg.coef_
                
    #    reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    #    reg.fit(A_encode,b)
    #    x1 = reg.coef_
    
    
        #######################################
        #x1 = np.linalg.lstsq(A_encode,b)[0]        
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
#        nTimes = 2
#        nGroups = 5
#        scores = []
#        for i in np.arange(nTimes):
#            #scores = []
#            kf = KFold(n_splits=nGroups,shuffle=True,random_state=3)
#            for train_index, test_index in kf.split(A_encode,b):
#                reg.fit(A_encode[train_index],b[train_index])
#                x1_reg = reg.coef_
#                
#                #difference = b - Ax
#                diff = b[test_index] - reg.predict(A_encode[test_index])
#                score = np.mean(np.square(diff))
#                #print "score: {}".format(score)
#                scores.append(score)
#        print "Average Score: {}".format(np.mean(scores))
#        CV_score_list.append(np.mean(scores))
        ######################################
        
        
    
        
        a = np.zeros(len(columns)+1)
        
        index_sector = columns.index(Sector)
        index_region = columns.index(Region)
        index_ImpliedRating = columns.index(Implied_Rating)
        index_Global = len(columns)
        index = [index_sector, index_region, index_ImpliedRating, index_Global]
        for i in range(len(index)):
            a[index] = 1
        estimated_spreadz = np.exp(np.dot(a,x1))
        #estimated_spreadz = np.dot(a,x1)
        
        dataframe_x = dataframe_x.append(pd.DataFrame(x1).T)
        #dataframe_x = dataframe_x.append(pd.DataFrame(np.exp(x1)).T)
        container.append(estimated_spreadz)
        
        
    ################### Nelson Siegel Fitting Area ################################
    lambda_ = 0.0609
    
    A = zip(time_to_maturity,container)
    for i, (x,y) in enumerate(A):
        #print x
        A[i] =  A[i] + (1,short(lambda_,x),medium(lambda_,x))
    dd = pd.DataFrame(A)
    dd.dropna(inplace = True)
    b = dd.iloc[:,1].values
    W = dd.iloc[:,2:].values
               
    reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    reg.fit(W,b)
    NS_coef = reg.coef_

    NS_Curve = [NS(NS_coef[0],NS_coef[1],NS_coef[2],lambda_,o) for o in time_to_maturity]
        
        
    NS_Curve.append(original_df[original_df.Ticker == tickerName]["Recovery"].values[0]) 
    container.append(original_df[original_df.Ticker == tickerName]["Recovery"].values[0])
    
    #done for all tenors
    cont = pd.DataFrame(container).T
    spread_list = pd.concat([spread_list,cont.T],axis=1)


    cont_NS = pd.DataFrame(NS_Curve).T
    spread_list_NS = pd.concat([spread_list_NS,cont_NS.T],axis=1)
    
    #print "Final CV Score: {}".format(np.mean(CV_score_list))
    
#    cont = pd.DataFrame([original_spreads,container]).T
#    cont.columns = ["Original","Model2"]
#    cont.index = columnnames
#    
#    cont = pd.concat([cont,myset_data],axis=1)
    
    
#    columns.append("Global")
#    dataframe_x = dataframe_x.T
#    dataframe_x.index = columns
#    dataframe_x.columns = columnnames
    

spread_list.columns = ticker_list
columnnames = columnnames.append(pd.Index(["Recovery"]))
spread_list.index = columnnames
spread_list.T.to_excel(location + "Model2_Financials.xlsx")

spread_list_NS.columns = ticker_list
spread_list_NS.index = columnnames
spread_list_NS.T.to_excel(location + "Model2+NS_Financials.xlsx")