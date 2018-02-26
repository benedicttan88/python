# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:45:39 2017

Hagan LogNormal
Hagan Normal




@author: WB512563
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def haganLogNormalApprox (y, expiry , F_0 , alpha_0 , beta ,nu , rho ):
    """
    Function which returns the Black implied volatility ,
    computed using the Hagan et al. lognormal
    approximation .
    @var y: option strike
    @var expiry: option expiry (in years)
    @var F_0: forward interest rate
    @var alpha_0: SABR Alpha at t=0
    @var beta : SABR Beta
    @var rho: SABR Rho
    @var nu: SABR Nu
    """
    one_beta = 1.0 - beta
    one_betasqr = one_beta * one_beta
    if F_0 != y:
        fK = F_0 * y
        fK_beta = math.pow(fK , one_beta / 2.0)
        log_fK = math.log(F_0 / y)
        z = nu / alpha_0 * fK_beta * log_fK
        x = math.log(( math .sqrt (1.0 - 2.0 * rho * z + z * z) + z - rho) / (1 - rho))
        sigma_l = (alpha_0 / fK_beta / (1.0 + one_betasqr / 24.0 * log_fK * log_fK + math.pow( one_beta * log_fK , 4) / 1920.0) * (z / x))
        sigma_exp = ( one_betasqr / 24.0 * alpha_0 * alpha_0 / fK_beta / fK_beta + 0.25 * rho * beta * nu * alpha_0 / fK_beta + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu)
        sigma = sigma_l * ( 1.0 + sigma_exp * expiry)
    else:
        f_beta = math.pow(F_0 , one_beta)
        f_two_beta = math.pow(F_0 , (2.0 - 2.0 * beta ))
        sigma = (( alpha_0 / f_beta) * (1.0 + (( one_betasqr / 24.0) * ( alpha_0 * alpha_0 / f_two_beta ) + (0.25 * rho * beta * nu * alpha_0 / f_beta) + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu) * expiry))
    
    return sigma



def drawTwoRandomNumbers(rho):
    """
    Draw a pair of correlated random numbers.
    SABR Model 81
    @var rho: SABR Rho
    """
    rand_list = []
    z1 = random.gauss(0,1)
    y1 = random.gauss(0,1)
    rand_list.append(z1)
    term1 = z1 * rho
    term2 = (y1 * math.pow ((1.0 - math.pow(rho , 2.0)) , 0.5))
    x2 = term1 + term2
    rand_list.append(x2)
    
    return rand_list

def drawNrandomNumebrs(correlation_matrix):
    
    Lower_tri = np.linalg.cholesky(correlation_matrix);
    
    size = correlation_matrix.shape[0]
    uncorrelated_var = np.random.randn(size);
    
    correlated_variables = np.dot(Lower_tri,uncorrelated_var);
                                 
    return correlated_variables;
    



def simulateSABRMonteCarloEuler (no_of_sim , no_of_steps , expiry , F_0 , alpha_0 , beta , rho , nu):
    """
    Monte Carlo SABR using Euler scheme.
    @var no_of_sim : Monte Carlo paths
    @var no_of_steps : discretization steps required
    to reach the option expiry date
    @var expiry: option expiry (in years)
    @var F_0: forward interest rate
    @var alpha_0: SABR Alpha at t=0
    @var beta : SABR Beta
    @var rho: SABR Rho
    @var nu: SABR Nu
    """
    
    # Step length in years
    dt = float(expiry) / float( no_of_steps )
    dt_sqrt = math.sqrt(dt)
    no_of_sim_counter = 0
    simulated_forwards = []
    
    mylist = []
    
    while no_of_sim_counter < no_of_sim :
        
        F_t = F_0
        alpha_t = alpha_0
        no_of_steps_counter = 1

            
        
        while no_of_steps_counter <= no_of_steps :
            # Zero absorbing boundary used for all the beta
            # choices except beta = 0 and beta = 1
            if (( beta > 0 and beta < 1) and F_t <= 0):
                F_t = 0
                if no_of_sim_counter ==1:
                    mylist.append(F_t);
                no_of_steps_counter = no_of_steps + 1
            else:
                # Generate two correlated random numbers
                rand = drawTwoRandomNumbers(rho)
                # Simulate the forward interest rate using the Euler scheme. Use the absolute for the diffusion to avoid numerical issues if the forward interest rate goes into negative territory
                dW_F = dt_sqrt * rand[0]
                F_b = math.pow(abs(F_t), beta )
                F_t = F_t + alpha_t * F_b * dW_F
                if no_of_sim_counter ==1:
                    mylist.append(F_t);
                # Simulate the stochastic volatility using the Euler scheme.
                dW_a = dt_sqrt * rand[1]
                alpha_t = (alpha_t + nu * alpha_t * dW_a )
                
            no_of_steps_counter += 1
            
        # At the end of each path , we store the forward
        # interest rate in a list
        simulated_forwards.append(F_t)
        no_of_sim_counter = no_of_sim_counter + 1
        
    return simulated_forwards, mylist



if __name__ == "__main__":
    
    ############
    ## SABR LOGNORMAL VOL
    ############
    
    nu = 0.03
    beta = 0.9
    rho = -0.04
    alpha = 0.01
    
    f = 0.3
    option_strike = 0.5;
    option_expiry = 3.6;
    
    print haganLogNormalApprox(option_strike,option_expiry,f, alpha, beta, nu, rho)

    
    
    strike_list = np.linspace(0.01,0.1,100);
    paramters_list = "beta = {}, alpha ={} , rho = {}, nu ={}, f= {}".format(beta,alpha,rho,nu,f)
    plt.xlabel("Strike")
    plt.ylabel("Implied Vol")
    plt.plot(strike_list, [haganLogNormalApprox(strike,option_expiry, f, alpha, beta, nu, rho) for strike in strike_list], label=paramters_list)
    
    
    
    #########
    ## CORRELATED VAR
    #########
    
    rho = 0.2;
    print(drawTwoRandomNumbers(0.2));
                              
    
    
    #################################
    ####    SABR Evolution
    #################################
    num_simulations = 1000;
    time_steps = 1000
    years = 15
    
    f0 = 0.025
    distribution_rates , sample_path = simulateSABRMonteCarloEuler(num_simulations,time_steps,years,f0,alpha*1.7,beta,rho,nu)
    
    
    
    
    
    
    
    
    