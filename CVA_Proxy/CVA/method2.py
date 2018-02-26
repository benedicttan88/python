# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:14:05 2017

@author: WB512563
"""

import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.model_selection import KFold

import sklearn.svm

def cva_proxy(df,TickerName,doKFold=False):
    
    
    #drop all the Blanks out of the eequation

    
    ###################################
    #### CHOOSE COUNTRY
    ###################################
    
    if (df[df.Ticker == TickerName].empty):
        print "CDS NAME NOT FOUND IN YOUR SET"
    else:
        
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
        
    ## total number of tables to make: 11
    # 8 to 19
    columnnames = df.columns[7:18]
    Tenors = columnnames.str[6:].tolist()
    
    
    dataframe_x = pd.DataFrame()
    container = []          #craete container to store items
    
    for i in np.arange(11):
        
        df.dropna(subset=["Spread"+Tenors[i]],inplace=True)
                
        
        print i
        print "Tenor: {}".format(Tenors[i])
        #set b
        print "b: " , columnnames[i]
        b = np.log(df[columnnames[i]].values)
        #set A
        A = df[["Sector","Region","ImpliedRating"]]
        #A = df[["Region","ImpliedRating"]]
        encoded_A = pd.get_dummies(A)
        columns = encoded_A.columns.tolist()
        
        A_encode = encoded_A.values
        #A_encode = np.column_stack( (A_encode,np.ones(A_encode.shape[0]) ))
        
        ###########################################
        ######## 2 norm 
        ###########################################
        reg = sklearn.linear_model.Ridge(alpha=0.5)
        reg.fit(A_encode,b)
        x1 = reg.coef_
        x1 = np.append(x1,reg.intercept_)
        
        
        SVR = sklearn.svm.LinearSVR(fit_intercept=False)
        #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        #svr_lin = SVR(kernel='linear', C=1e3)
        #svr_lin.fit(A_encode, b)
        SVR.fit(A_encode,b)

        #print len(x2)
        
        
        ######################################


        ####################################    
        ## ----- CV AREA
        if (doKFold == True):
            scores = []
            kf = KFold(n_splits=10)
            for train_index, test_index in kf.split(A_encode,b):
                reg.fit(A_encode[train_index],b[train_index])
                score = reg.score(A_encode[test_index],b[test_index])
                print "score: {}".format(score)
                scores.append(score)
            print "Average Score: {}".format(np.mean(scores))
        ####################################


                
        ######################################
        ##### Regenerate your a
        ######################################
        a = np.zeros(len(columns)+1)
        
        index_sector = columns.index(Sector)
        index_region = columns.index(Region)
        index_ImpliedRating = columns.index(Implied_Rating)
        index_Global = len(columns)
        index = [index_sector, index_region, index_ImpliedRating, index_Global]
        for i in range(len(index)):
            a[index] = 1
             
        ######################################
             
        estimated_spreadz = np.exp(np.dot(a,x1))
        
        #dataframe_x = dataframe_x.append(pd.DataFrame(x1).T)
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
    
    
    return dataframe_x, \
            cont, \
            myset_names
    