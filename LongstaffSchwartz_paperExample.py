# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:34:28 2017

@author: tanbened
"""


import numpy as np


r = 0.06

v0_european = np.exp(-r*3)*(0.07+0.18+0.2+0.09)/8
v0_american = (np.exp(-r)*(0.17 + 0.34 + 0.18 + 0.22) + np.exp(-r*3)*0.07)/8


def f(x):

    return -1.070 + 2.983*x - 1.813*x*x


t3 = [3,4,6,7]
t2 = [4,6,7]
t1 = [4,6,7,8]
