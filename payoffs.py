# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 18:19:24 2017

@author: Benedict Tan
"""

import numpy as np
import matplotlib.pyplot as plt



def prdc_payoff(S_0,S_T,F,Margin,Floor,Cap):
    
    
    Notional = 1;
    
    inner = max((F*S_T/S_0)-Margin,Floor)
    outer = min(inner,Cap)
    
    return Notional*outer


s_0 = 120
Cap = 0.06
Floor = 0.0
F = 0.24
Margin = 0.18


t = np.linspace(80,130,51)
y = [prdc_payoff(s_0,i,F,Margin,Floor,Cap) for i in t]

plt.plot(t,y)
plt.title("PRDC Payoff")
plt.xlabel("S_T")
plt.ylabel("coupon rate")
