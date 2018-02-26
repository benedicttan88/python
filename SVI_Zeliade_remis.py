# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:56:43 2018

@author: tanbened
"""

import numpy as np
import scipy
import pandas as pd

from math import sqrt

#%%









#%%

import sys
def acceptable(S, a, d, c, vT):
    eps = sys.float_info.epsilon
    return -eps <= c and c <= 4 * S + eps and abs(d) <= min(c, 4 * S - c) + eps and -eps <= a and a <= vT.max() + eps


# The error function from the optimizer, in terms of a, d and c so we don't need to keep dividing by T:

# In[9]:

def sum_of_squares(x, a, d, c, vT):
    diff = a + d * x + c * np.sqrt(x * x + 1) - vT
    return (diff * diff).sum()


# The key part of the Zeliade paper - identify the values of `a`, `d` and `c` that produce the lowest sum-of-squares error when the gradient of the `f` function is zero, ensuring that the parameters remain in the domain `D` identified in the paper.

# In[10]:

def solve_grad(params, x, vT):
    
    
    S , M = params
    
    ys = (x - M) / S
    
    y = ys.sum()
    y2 = (ys * ys).sum()
    y2one = (ys * ys + 1).sum()
    ysqrt = np.sqrt(ys * ys + 1).sum()
    y2sqrt = (ys * np.sqrt(ys * ys + 1)).sum()
    v = vT.sum()
    vy = (vT * ys).sum()
    vsqrt = (vT * np.sqrt(ys * ys + 1)).sum()

    matrix = [
        [1, y, ysqrt],
        [y, y2, y2sqrt],
        [ysqrt, y2sqrt, y2one]
    ]
    vector = [v, vy, vsqrt]
    _a, _d, _c = np.linalg.solve(np.array(matrix), np.array(vector))

    if acceptable(S, _a, _d, _c, vT):
        return _a, _d, _c, sum_of_squares(ys, _a, _d, _c, vT)

    _a, _d, _c, _cost = None, None, None, None
    for matrix, vector, clamp_params in [
        ([[1,0,0],[y,y2,y2sqrt],[ysqrt,y2sqrt,y2one]], [0, vy, vsqrt], False), # a = 0
        ([[1,0,0],[y,y2,y2sqrt],[ysqrt,y2sqrt,y2one]], [vT.max(),vy,vsqrt], False), # a = _vT.max()

        ([[1,y,ysqrt],[0,-1,1],[ysqrt,y2sqrt,y2one]], [v,0,vsqrt], False), # d = c
        ([[1,y,ysqrt],[0,1,1],[ysqrt,y2sqrt,y2one]], [v,0,vsqrt], False), # d = -c 

        ([[1,y,ysqrt],[0,1,1],[ysqrt,y2sqrt,y2one]], [v,4*S,vsqrt], False), # d <= 4*s-c
        ([[1,y,ysqrt],[0,-1,1],[ysqrt,y2sqrt,y2one]], [v,4*S,vsqrt], False), # -d <= 4*s-c

        ([[1,y,ysqrt],[y,y2,y2sqrt],[0,0,1]], [v,vy,0], False), # c = 0
        ([[1,y,ysqrt],[y,y2,y2sqrt],[0,0,1]], [v,vy,4*S], False), # c = 4*S

        ([[1,y,ysqrt],[0,1,0],[0,0,1]], [v,0,0], True), # c = 0, implies d = 0, find optimal a
        ([[1,y,ysqrt],[0,1,0],[0,0,1]], [v,0,4*S ], True), # c = 4s, implied d = 0, find optimal a

        ([[1,0,0],[0,-1,1],[ysqrt,y2sqrt,y2one]], [0,0,vsqrt], True), # a = 0, d = c, find optimal c
        ([[1,0,0],[0,1,1],[ysqrt,y2sqrt,y2one]], [0,0,vsqrt], True), # a = 0, d = -c, find optimal c

        ([[1,0,0],[0,1,1],[ysqrt,y2sqrt,y2one]], [0,4*S,vsqrt], True), # a = 0, d = 4s-c, find optimal c
        ([[1,0,0],[0,-1,1],[ysqrt,y2sqrt,y2one]], [0,4*S,vsqrt], True) # a = 0, d = c-4s, find optimal c
    ]:
        a, d, c = np.linalg.solve(np.array(matrix), np.array(vector))
        
        if clamp_params:
            dmax = min(c, 4 * S - c)
            a = min(max(a, 0), vT.max())
            d = min(max(d, -dmax), dmax)
            c = min(max(c, 0), 4 * S)

        cost = sum_of_squares(ys, a, d, c, vT)
        
        if acceptable(S, a, d, c, vT) and (_cost is None or cost < _cost):
            _a, _d, _c, _cost = a, d, c, cost
        
    assert _cost is not None, "S=%s, M=%s" % (S, M)
    return _a, _d, _c, _cost


# Use scipy's optimizer to minimize the error function

# In[11]:

from scipy import optimize
def solve_grad_get_score(params, x, vT):
    return solve_grad(params, x, vT)[3]

def calibrate(df):
    vT = df.IMPLIEDVOL * df.IMPLIEDVOL * df.t
    res = optimize.minimize(solve_grad_get_score, [.1, .0], args=(df.MONEYNESS, vT), bounds=[(0.001, None), (None, None)])
    assert res.success
    S, M = res.x
    a, d, c, _ = solve_grad((S, M), df.MONEYNESS, vT)
    T = df.t.max() # should be the same for all rows
    A, P, B = a / T, d / c, c / (S * T)
    assert T >= 0 and S >= 0 and abs(P) <= 1
    return A, P, B, S, M


def calibrate_ben(marketvol , T , log_moneyness):
    vT = np.array([T*m*m for m in marketvol])
    
    res = optimize.minimize(solve_grad_get_score, [.1, .0], args=(log_moneyness, vT), bounds=[(0.001, None), (None, None)])
    assert res.success
    S, M = res.x
    
    params = S , M
    a, d, c, _ = solve_grad(params, log_moneyness, vT)
    #T = df.t.max() # should be the same for all rows
    
    A, P, B = a / T, d / c, c / (S * T)
    
    assert T >= 0 and S >= 0 and abs(P) <= 1
    return A, P, B, S, M



def calibrate_ben_reverse(marketvol, T , log_moneyness):
    
    vT = np.array([T*m*m for m in marketvol])
    
    S , M = 0.01,0.01
    params = S , M
    a, d, c, _ = solve_grad(params, log_moneyness, vT)
    
    res = optimize.minimize(solve_grad_get_score, [.1, .0], args=(log_moneyness, vT), bounds=[(0.001, None), (None, None)])
    assert res.success
    S, M = res.x
    
    a, d, c, _ = solve_grad(params, log_moneyness, vT)
    
    A, P, B = a / T, d / c, c / (S * T)
    
    assert T >= 0 and S >= 0 and abs(P) <= 1
    return A, P, B, S, M


# SVI formula for the volatility (not variance):

# In[12]:
def svi(A, P, B, S, M, T, x):
    return T*(A + B * (P * (x - M) + sqrt((x - M) * (x - M) + S * S)))


#%%


df = pd.read_csv("C:\\Users\\tanbened\\python\\Input\\SVI Interpolation Data\\spx_index_21sep.csv",header=None)

strikes = df.iloc[:,0]
marketvol = df.iloc[:,1]
marketvol = [i/100 for i in marketvol]

T = 0.693150685
#T = 1
F_0 = 2753.31

log_moneyness = np.array([np.log(x/F_0) for x in strikes])




A, P, B, S, M = calibrate_ben(marketvol, T, log_moneyness)


model_total_implied_variance = [svi(A,P,B,S,M,T, xx) for xx in log_moneyness]
market_total_implied_variance = [(T*m*m) for m in marketvol]

calibration_report = pd.DataFrame(zip(market_total_implied_variance,model_total_implied_variance),columns=["Market Variance","Model Variance"], index=log_moneyness)
calibration_report.plot()



calibrate_ben_reverse(marketvol, T, log_moneyness)



















