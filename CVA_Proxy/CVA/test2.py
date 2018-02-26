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


#---------------------------------------------------


#drop all the Blanks out of the eequation
#df.dropna(inplace = True)


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

outer_outer_scores = []
CV_score_list = []
dataframe_x = pd.DataFrame()
container = []          #craete container to store items

            
for i in np.arange(11):
    
    #Under TENOR I
    df.dropna(subset=["Spread"+Tenors[i]],inplace=True)
    df.dropna(subset=["ImpliedRating"],inplace=True)
    #drop the missing ones
    
    ######### Get A_encode , b ########################################
    original_b = np.log(df[columnnames[i]].values)
    A = df[["Sector","Region","ImpliedRating"]]
    encoded_A = pd.get_dummies(A)
    columns = encoded_A.columns.tolist()
    
    A_encode = encoded_A.values
    A_encode = np.column_stack( (A_encode,np.ones(A_encode.shape[0]) ))
    ###################################################################
    
    nGroups = 5


    kf = KFold(n_splits=nGroups,shuffle=True,random_state=3)
    for train_index, test_index in kf.split(A_encode,original_b):
        
        #only interested in test index
        inner_score = []
        #collect tickers for the test set
        for j in test_index:
            #for each ticker in the test set:
            #go tenor by tenor
            
            #get Sector, Region, ImpliedRating
            Sector = "Sector_" + A.iloc[j,:].Sector
            Region = "Region_" + A.iloc[j,:].Region                                       
            Implied_Rating = "ImpliedRating_" + str(A.iloc[j,:].ImpliedRating);
            
            container = []          #craete container to store items
            for k in np.arange(11):
                #drop
                
                df.dropna(subset=["Spread"+Tenors[k]],inplace=True)
                df.dropna(subset=["ImpliedRating"],inplace=True)

                temp_b = np.log(df[columnnames[k]].values)
                temp_A = df[["Sector","Region","ImpliedRating"]]
                temp_encoded_A = pd.get_dummies(temp_A)
                columns = temp_encoded_A.columns.tolist()
                
                temp_A_encode = temp_encoded_A.values
                temp_A_encode = np.column_stack( (temp_A_encode,np.ones(temp_A_encode.shape[0]) ))             #Add the Global ones
                
                ## 2 norm ###################################################
#                reg = sklearn.linear_model.Ridge(alpha=2.0,fit_intercept=False)
#                reg.fit(temp_A_encode,temp_b)
#                x1 = reg.coef_
                
                
                reg = sklearn.linear_model.Lasso(fit_intercept=False,alpha=0.01)
                reg.fit(temp_A_encode,temp_b)
                x1 = reg.coef_

                #############################################################
                
                a = np.zeros(len(columns)+1)
                
                index_sector = columns.index(Sector)
                index_region = columns.index(Region)
                index_ImpliedRating = columns.index(Implied_Rating)
                index_Global = len(columns)
                index = [index_sector, index_region, index_ImpliedRating, index_Global]
                for w in range(len(index)):
                    a[index] = 1
                estimated_spreadz = np.exp(np.dot(a,x1))
                #estimated_spreadz = np.dot(a,x1)
                
                #dataframe_x = dataframe_x.append(pd.DataFrame(x1).T)

                container.append(estimated_spreadz)

            
            #apply smoothing once i finish regression
            ################### Nelson Siegel Fitting Area ################################
            lambda_ = 0.0609
            
            temp_NS= zip(time_to_maturity,container)
            for p, (x,y) in enumerate(temp_NS):
                #print x
                temp_NS[p] =  temp_NS[p] + (1,short(lambda_,x),medium(lambda_,x))
            dd = pd.DataFrame(temp_NS)
            dd.dropna(inplace = True)
            b = dd.iloc[:,1].values
            W = dd.iloc[:,2:].values
                       
            reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
            reg.fit(W,b)
            NS_coef = reg.coef_
            #print NS_coef
            NS_Curve = [NS(NS_coef[0],NS_coef[1],NS_coef[2],lambda_,o) for o in time_to_maturity]
                
            ## Get the difference finally
            Loss = np.square(np.exp(original_b[j]) - NS_Curve[i])
            inner_score.append(Loss)
            
        outer_outer_scores.append(np.mean(inner_score))    

        print "Fold Done"
        #outer_outer_scores.append(np.mean(inner_score))        
        print "Tenor {} Done".format(i)
        










#
#cont = pd.DataFrame([original_spreads,container,NS_Curve]).T
#cont.columns = ["Original","Model2","Model2(NS)"]
#cont.index = columnnames
#
#cont = pd.concat([cont,myset_data],axis=1)
#
#
#columns.append("Global")
#dataframe_x = dataframe_x.T
#dataframe_x.index = columns
#dataframe_x.columns = columnnames
#
#
#ax = cont.plot(title="CDS Spreads of Similar Sector, Region, ImpliedRating\n Sector:{} Region: {} , Rating: {}".format(c["Sector"].values[0],c["Region"].values[0],c["ImpliedRating"].values[0]))
#ax.lines[1].set_linewidth(5)
#ax.lines[2].set_linewidth(5)
#ax.lines[3].set_linewidth(5)
#ax.set_ylabel("CDS Spread")
#ax.set_xlabel("Tenor")


