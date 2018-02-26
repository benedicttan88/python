# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:24:41 2017

@author: WB512563
"""


import pandas as pd
import numpy as np
import pickle


def convert_ImpliedRating_toOrdinal(df):

    #df.ix[df.ImpliedRating == "AAA" ,"ImpliedRating"] = 0;
    df.ix[df.ImpliedRating == "AA" ,"ImpliedRating"] = 1
    df.ix[df.ImpliedRating == "A" ,"ImpliedRating"] = 2
    df.ix[df.ImpliedRating == "BBB" ,"ImpliedRating"] = 3
    df.ix[df.ImpliedRating == "BB" ,"ImpliedRating"] = 4
    df.ix[df.ImpliedRating == "B" ,"ImpliedRating"] = 5
    df.ix[df.ImpliedRating == "CCC" ,"ImpliedRating"] = 6
    df.ix[df.ImpliedRating == "D" ,"ImpliedRating"] = 7

def convert_Single_ImpliedRating_toOrdinal(Rating):
    """
    Rating is a text. e.g "AAA" -> 6
    """
    if (Rating == "AAA"):
        ordinal_rating = 0
    elif(Rating == "AA"):
        ordinal_rating = 1
    elif(Rating == "A"):
        ordinal_rating = 2        
    elif(Rating == "BBB"):
        ordinal_rating = 3        
    elif(Rating == "BB"):
        ordinal_rating = 4        
    elif(Rating == "B"):
        ordinal_rating = 5        
    elif(Rating == "CCC"):
        ordinal_rating = 6
    elif(Rating == "D"):
        ordinal_rating = 7
        
    return ordinal_rating

def gower_distance(myProposedType,d,type_D):
    
    
    if (type_D == "Categorical"):
        s = []
        for i in d:
            if (myProposedType == i):
                s.append(1);
            else:
                s.append(0);
                        
    elif(type_D == "Ordinal"):
        s = []
        j = myProposedType
        T_i_j = d.count(j)
        
        T_i_min = d.count(min(d))
        T_i_max = d.count(max(d))
        
        R_i_min = min(d)
        R_i_max = max(d)
        for k in d:
            T_i_k = d.count(k)
            top = abs(j-k) - 0.5*(T_i_j - 1) - 0.5*(T_i_k - 1)
            bottom = R_i_max - R_i_min - 0.5*(T_i_max - 1) - 0.5*(T_i_min - 1)
            temp_s = top/bottom
            s.append(1-temp_s)
            
        
#    elif(type_D == "Numeric"):
    
    return s    


location = "D:\\python\\CVA_Proxy\\Input\\"
output_location = "D:\\python\\CVA_Proxy\\Output\\"


#Read in the file


location = "D:\\python\\CVA_Proxy\\Input\\"
pkl_file = open(location + "myfile3.pkl", 'rb')
df = pickle.load(pkl_file)
pkl_file.close()
df = df.reset_index(drop=True)
df.dropna(subset=["ImpliedRating"],inplace=True)

#map to the 


A = df[["Ticker","Country","Sector","Region","ImpliedRating"]].copy()
A = A.reset_index(drop=True)
convert_ImpliedRating_toOrdinal(A)


columnsA = A.columns.tolist()
TypeA = ["Categorical","Categorical","Ordinal"]
A_ZIP = zip(columnsA,TypeA)

myProposedType = ["Financials","N.Amer","AA"]


            
for name,values in A.iteritems():
    
    
    if name == "Sector":
        
        d = values
        gow_sector = gower_distance(myProposedType[0],d,"Categorical")
    
    elif(name == "Region"):
        d = values
        gow_region = gower_distance(myProposedType[1],d,"Categorical")

    elif(name == "ImpliedRating"):

        d = values.tolist()
        myOrdinalRating = convert_Single_ImpliedRating_toOrdinal(myProposedType[2])
        gow_ImpliedRating = gower_distance(myOrdinalRating,d,"Ordinal")
    
    
weight_Sector = 0.3
weight_Region = 0.3
weight_ImpliedRating = 1 - weight_Sector - weight_Region
weights = [weight_Sector, weight_Region, weight_ImpliedRating]

t1 = weight_Sector*np.array(gow_sector)
t2 = weight_Region*np.array(gow_region)
t3 = weight_ImpliedRating*np.array(gow_ImpliedRating)

dd = pd.DataFrame([t1,t2,t3]).T

sum_ = pd.DataFrame(dd.sum(axis=1))

A["Gower_Distance"]= sum_
 
                 
    
    
    
    
    