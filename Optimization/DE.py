# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 11:36:09 2017

@author: Benedict Tan
"""

import numpy as np
import pandas as pd



def differential_evolution(f, D_dimension, N_samples, min_bound, max_bound):
    """
    D_Dimension = number of dimension for each vector
    N_samples = generate N random samples within the sample space [min, max]
    """
    #Multidimensional of D dimensions
    #in our case D= 3 for CIR Model
    
    Min_Bound = min_bound
    Max_Bound = max_bound
    
    #restrict the space to only positive doubles in (0,1)
    D = D_dimension
    N = N_samples*D
    #N=5
    
    CR = 0.9   #[0,1]
    F = 0.8    #(0,2)
    
    #generate N points
    generation_ = 0
    matrix = np.random.random((D,N))*Max_Bound
    
    full_list = np.arange(N)
    collecting_fitness_scores = np.zeros(N)
    while generation_ < 200:
        
        #print "\nAt Generation {}".format(generation_)
        
        for i in range(matrix.shape[1]):        #going across columns...N 
            
            #print "at column {}".format(i)
            
            #randomly pick 3 indices from the subset list
            new_list = np.delete(full_list,i)
            x1,x2,x3 = np.random.choice(new_list,3)
            
            #Mutation Stage
            V = matrix[:,x1] + F * (matrix[:,x2] - matrix[:,x3])            #check bounds here
            V = checkBounds(V,Min_Bound,Max_Bound)
                      
                      
            U = np.zeros(len(V))                #make trial vector
            random_choice = np.random.choice(D)
            #Recombination Stage (Crossover) Forming Trial Vector
            for j in range(len(V)):
                s = np.random.uniform()         #draw random uniform number in [0,1]
                if (s <= CR) or (j == random_choice):                     #if s <= CR then crossover
                    U[j] = V[j]
                else:
                    U[j] = matrix[j,i]
            #U[0] = V[0]                         #definite crossover
            
#            #Comparing Fitness Scores
#            fitness_trial_vec = f(U)
#            fitness_original = f(matrix[:,i])        
#            if fitness_trial_vec <= fitness_original:
#                matrix[:,i] = U
#                collecting_fitness_scores[i] = fitness_trial_vec
#            else:
#                collecting_fitness_scores[i] = fitness_original
            
            x1_satisfied = checkConditions(U)
            x2_satisfied = checkConditions(matrix[:,i])
            
            fitness_trial_vec = f(U)
            fitness_original = f(matrix[:,i])
            
            
            if x1_satisfied and x2_satisfied:
                if fitness_trial_vec < fitness_original:
                    matrix[:,i] = U
                    collecting_fitness_scores[i] = fitness_trial_vec
                else:
                    collecting_fitness_scores[i] = fitness_original
            elif (x1_satisfied == True) and (x2_satisfied == False):
                matrix[:,i] = U
                collecting_fitness_scores[i] = fitness_trial_vec
            elif (x1_satisfied == False) and (x2_satisfied == True):
                collecting_fitness_scores[i] = fitness_original
            else:
                if fitness_trial_vec < fitness_original:
                    matrix[:,i] = U
                    collecting_fitness_scores[i] = fitness_trial_vec
                else:
                    collecting_fitness_scores[i] = fitness_original
            
            
            
            #print collecting_fitness_scores
            
        
        generation_ += 1
    
    print "smallest fitness score: {}".format(np.min(collecting_fitness_scores))
    smallest = matrix[:,np.argmin(collecting_fitness_scores)]
    print "smallest: {}".format(smallest)
    
    return smallest




def checkBounds(vec,min_bound,max_bound):
    
    for i in range(len(vec)):
        
        if vec[i] < min_bound:
            vec[i] = min_bound
        if vec[i] > max_bound:
            vec[i] = max_bound
               
    return vec


def checkConditions(x):
    
    condition = 2 * x[0] * x[1] - x[2]*x[2]
    if condition > 0:
        return True
    else:
        return False
    
    
def compare(x1,x2,f):
    
    x1_satisfied = checkConditions(x1)
    x2_satisfied = checkConditions(x2)
    
    if x1_satisfied and x2_satisfied:
        if f(x1) < f(x2):
            return f(x1)
        else:
            return f(x2)
    elif (x1_satisfied == True) and (x2_satisfied == False):
        return f(x1)
    elif (x1_satisfied == False) and (x2_satisfied == True):
        return f(x2)
    else:
        if f(x1) < f(x2):
            return f(x1)
        else:
            return f(x2)
        
        
        
        
    
    
    




