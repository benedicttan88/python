# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 16:42:00 2017

@author: WB512563
"""
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot(df,order):
    
    fig, axs = plt.subplots(nrows=2,ncols=3)
    sns.boxplot(x="ImpliedRating",y="Spread6m",data=df,order=order,ax=axs[0,0]).set_title("Spread6m")
    sns.boxplot(x="ImpliedRating",y="Spread1y",data=df,order=order,ax=axs[0,1]).set_title("Spread1y")
    sns.boxplot(x="ImpliedRating",y="Spread2y",data=df,order=order,ax=axs[0,2]).set_title("Spread2y")
    sns.boxplot(x="ImpliedRating",y="Spread3y",data=df,order=order,ax=axs[1,0]).set_title("Spread3y")
    sns.boxplot(x="ImpliedRating",y="Spread4y",data=df,order=order,ax=axs[1,1]).set_title("Spread4y")
    sns.boxplot(x="ImpliedRating",y="Spread5y",data=df,order=order,ax=axs[1,2]).set_title("Spread5y")

    for i, ax in enumerate(fig.axes):
        ax.set_xlabel('')
        ax.set_ylabel('')
    fig.text(0.08, 0.7, 'CDS Spread', ha='center', va='center', rotation='vertical')
    fig.text(0.08, 0.35, 'CDS Spread', ha='center', va='center', rotation='vertical')    
        
    fig, axs = plt.subplots(nrows=2,ncols=3)
    sns.boxplot(x="ImpliedRating",y="Spread7y",data=df,order=order,ax=axs[0,0]).set_title("Spread7y")
    sns.boxplot(x="ImpliedRating",y="Spread10y",data=df,order=order,ax=axs[0,1]).set_title("Spread10y")
    sns.boxplot(x="ImpliedRating",y="Spread15y",data=df,order=order,ax=axs[0,2]).set_title("Spread15y")
    sns.boxplot(x="ImpliedRating",y="Spread20y",data=df,order=order,ax=axs[1,0]).set_title("Spread20y")
    sns.boxplot(x="ImpliedRating",y="Spread30y",data=df,order=order,ax=axs[1,1]).set_title("Spread30y")

    for i, ax in enumerate(fig.axes):
        ax.set_xlabel('')
        ax.set_ylabel('')
    fig.text(0.08, 0.7, 'CDS Spread', ha='center', va='center', rotation='vertical')
    fig.text(0.08, 0.3, 'CDS Spread', ha='center', va='center', rotation='vertical') 
        
        
def boxplot_Sector(df):
    
    title= "Boxplot of 5Y CDS Spreads by Sector"
    fig, axs = plt.subplots()
    sns.boxplot(x="Sector",y="Spread5y",data=df).set_title(title)


def boxplot_Region(df):
    
    title= "Boxplot of 5Y CDS Spreads by Region"
    fig, axs = plt.subplots()
    sns.boxplot(x="Region",y="Spread5y",data=df).set_title(title)



def barplot(df):
    
    title= "Barplot of Number of CDS by ImpliedRating"
    fig, axs = plt.subplots()
    sns.barplot(x=df.index, y="Spread5y", data=df).set_title(title)
    axs.set_ylabel('Count of CDS Contracts')
    
    