# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 10:24:51 2017

@author: WB512563
"""
import sklearn
import pandas as pd
import numpy as np
import sklearn.linear_model
#######################
"""
Final use Method 1(B) described in paper (l2 regression)
"""
#######################

TickerName = "BRAZIL"

#######################
###     Settings
#######################

location = "Y:\\Ben\\CVA_Proxy\\Input\\"
filename = "V5 CDS Composites by Convention_26 Sep 17"


df = pd.read_excel(location + filename + ".xlsx",parse_cols="B:T,AH:AL")
#df = pd.read_csv(location + filename2 + ".csv",skiprows=2)

df = df[(df.Tier == "SNRFOR") & (df.Ccy == "USD")]

OTHER = df[(df.Sector == "Financials") | (df.Sector == "Government")]
OTHER_CRVES = df[~df.index.isin(OTHER.index)]
df.ix[OTHER_CRVES.index, "Sector"] = "Others"


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

#######################################
###     Iterating by Tenors
#######################################

for i in np.arange(11):
    
    df.dropna(subset=["Spread"+Tenors[i]],inplace=True)
    df.dropna(subset=["ImpliedRating"],inplace=True)

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

    ################
    ## Start of l2 regression
    ################
    reg = sklearn.linear_model.Ridge(alpha=2.0,fit_intercept=False)
    reg.fit(A_encode,b)
    x1 = reg.coef_
    ################
    ################
    
    
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


