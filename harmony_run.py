# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:43:27 2018

@author: [TWkiwi] Maxwell Chen
"""
import numpy as np
from HarmonyCore import HarmonyCore
class objective_function:
    def __init__(self,
                 input_X, 
                 input_Y,
                 iteration = 10000,
                 sample_size = -1, 
                 hmcr_proba = 0.7, 
                 par_proba = 0.3, 
                 adju_proba = 0.5,
                 harmony_menmory_size = 50,
                 up_down_limit = None):

        self.input_X = input_X
        self.input_Y = input_Y
        self.iteration = iteration
        if sample_size == -1:
            self.sample_size = len(input_X)
        else:
            self.sample_size = sample_size
        self.hmcr_proba = hmcr_proba
        self.par_proba = par_proba
        self.adju_proba = adju_proba
        self.vector_size = len(input_X[0])
        self.harmony_menmory_size = harmony_menmory_size
        if up_down_limit == None:
            self.up_down_limit = [[0,1]] * len(input_X[0])
        else:
            self.up_down_limit = up_down_limit
        



    '''
    Notic!!
    My fitness func was DEFINE AS AVERAGE ERROR FITNESS.
    The input_X was average vector. And input_Y was a double number.
    And weight is a new weight vector.

    You can customer yourself.
    '''
    def fitness(self,weight):
        e = 0.0
        input_X = np.mean(np.random.permutation(self.input_X)[:self.sample_size],axis=0)
        input_Y = np.mean(np.random.permutation(self.input_Y)[:self.sample_size],axis=0)
        weight = [float(i)/sum(weight) for i in weight]
        e += sum(np.multiply(input_X,weight)) - input_Y
        return abs(e)
    

if __name__ == '__main__':


    '''
    Load your data.(input_X & input_Y)
    input_X & input_Y both are matrix like this:

    input_X:
    [[ 0.123, 0.543, 0.87,-0.01 ],
     [ 0.23 , 0.4  , 0.87,-0.01 ],
     [-0.13 , 0.3  , 0.8 , 0.81 ],
     [-0.53 , 0.3  , 0.18, 0.1  ]]

    input_Y:
    [[1],
     [1],
     [1],
     [0]]
    ''' 
    input_X = [[ 0.123, 0.543, 0.87,-0.01 ],
     [ 0.23 , 0.4  , 0.87,-0.01 ],
     [-0.13 , 0.3  , 0.8 , 0.81 ],
     [-0.53 , 0.3  , 0.18, 0.1  ]]
    
    input_Y = [[1],
     [1],
     [1],
     [0]]

    '''
    init interface obj
    objective_function = objective_function("here your input_X","here your input_Y")
    here you can customer your parameter or not ( run as define ).
    For example:
    up_down_limit = [[-1,1],[-1,1],[-1,1],[-1,1]]
    '''
    up_down_limit = [[-1,1],[-1,1],[-1,1],[-1,1]]
    objective_function = objective_function(input_X,input_Y,sample_size = 2,up_down_limit=up_down_limit)
    

    '''
    pass your interface obj into a harmony core obj. 
    hs = HarmonyCore("here your interface obj")
    '''
    hs = HarmonyCore(objective_function)

    '''
    run harmony
    hs.run()
    '''
    hmm_vector,hmm_err_list,err_idx = hs.run()

    '''
    then you can get (hmm_vector,hmm_err_list,err_idx) after hs return.
    '''

    print('HMM_Vector:',hmm_vector)
    print('Best HMM_Vector:',hmm_vector[err_idx])
    print('hmm_err_list:',hmm_err_list) 
    print('err_index:',err_idx,'hmm_err:',hmm_err_list[err_idx]) 
          

    
