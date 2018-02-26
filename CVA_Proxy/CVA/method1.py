# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:14:05 2017

@author: WB512563
"""

import numpy as np
import pandas as pd
import sklearn.linear_model

from sklearn.model_selection import KFold

def cva_proxy(df,TickerName,doKFold=False):
    
    
    #drop all the Blanks out of the eequation
    df.drop("AvRating",axis=1,inplace= True)
    df.dropna(inplace = True)
    
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
    
    
    ## total number of tables to make: 11
    # 8 to 19
    columnnames = df.columns[7:18]
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
        A = df[["Sector","Region","ImpliedRating"]]
        #A = df[["Region","ImpliedRating"]]
        encoded_A = pd.get_dummies(A)
        columns = encoded_A.columns.tolist()
        
        A_encode = encoded_A.values
        A_encode = np.column_stack( (A_encode,np.ones(A_encode.shape[0]) ))
        
        print A_encode.shape
        #######################################
        
        ### 2 norm
#        reg = sklearn.linear_model.Ridge(alpha=1e-3)
#        reg.fit(A_encode,b)
#        x1 = reg.coef_
#        x1 = np.append(x1,reg.intercept_)
        reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
        reg.fit(A_encode,b)
        x1_scikit = reg.coef_
    
        x1 = x1_scikit

        #######################################
        
        #x1 = np.linalg.lstsq(A_encode,b)[0]
#        
#        diff =  x1 - x1_scikit
#        print "diff: {}".format(diff)
        
        #print "lstsq: {} , scikit: {}".format(x1,x1_scikit)
        #difference2 = b - np.dot(A_encode,x1)
        #d2 = np.linalg.norm(difference2)
        #print "||difference|| : ", d2
        
        
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
        
        a = np.zeros(len(columns)+1)
        
        index_sector = columns.index(Sector)
        index_region = columns.index(Region)
        index_ImpliedRating = columns.index(Implied_Rating)
        index_Global = len(columns)
        index = [index_sector, index_region, index_ImpliedRating, index_Global]
        for i in range(len(index)):
            a[index] = 1
        estimated_spreadz = np.exp(np.dot(a,x1))
        
        dataframe_x = dataframe_x.append(pd.DataFrame(x1).T)
        #dataframe_x = dataframe_x.append(pd.DataFrame(np.exp(x1)).T)
        container.append(estimated_spreadz)
        
        
        
    
    
    
    
    cont = pd.DataFrame([original_spreads,container]).T
    cont.columns = ["Original","LstSq"]
    cont.index = columnnames
    
    cont = pd.concat([cont,myset_data],axis=1)
    
    
    columns.append("Global")
    dataframe_x = dataframe_x.T
    dataframe_x.index = columns
    dataframe_x.columns = columnnames
    
    
    return dataframe_x,\
            cont,\
            myset_names
    