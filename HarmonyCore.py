# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:11:19 2018

@author: [TWkiwi] Maxwell Chen
"""

import numpy as np
class HarmonyCore(object):
    
    def __init__(self,harmony_obj):
        self.obj_func = harmony_obj
        self.hmm_matrix = list()
        matrix = []
        for limit in self.obj_func.up_down_limit:
            row = np.random.uniform(low=limit[0], high=limit[1], size=(1,self.obj_func.harmony_menmory_size))[0] 
            matrix.append(row)
        self.hmm_matrix = np.asarray(matrix).transpose()
    def run(self):
        hmm_err_list = [0] * len(self.hmm_matrix)
        for m_i in range(len(self.hmm_matrix)):
            vetor_list = self.hmm_matrix[m_i]

            error = self.obj_func.fitness(vetor_list)
            hmm_err_list[m_i] = error
            
        for itera in range(self.obj_func.iteration):
            vetor_list = [0] * self.obj_func.vector_size
            while True:
                for i in range(self.obj_func.vector_size):
                    if np.random.rand(1,)[0] < self.obj_func.hmcr_proba:
                        #new_vactor = 0.0
                        new_vactor = self.hmm_matrix[np.random.randint(self.obj_func.harmony_menmory_size, size=1)[0]][i]
                        if np.random.rand(1,)[0] < self.obj_func.par_proba:
                            if np.random.rand(1,)[0] < self.obj_func.adju_proba:
                                #new_vactor -= np.std(self.hmm_matrix[:][i]) * np.random.rand(1,)[0]
                                new_vactor -= (new_vactor - self.obj_func.up_down_limit[i][0]) * np.random.rand(1,)[0]
                            else:
                                #new_vactor += np.std(self.hmm_matrix[:][i]) * np.random.rand(1,)[0]
                                new_vactor += (self.obj_func.up_down_limit[i][0] - new_vactor) * np.random.rand(1,)[0]
                        vetor_list[i] = round(new_vactor, 3)
                    else: 
                        new_vactor = np.random.uniform(low=self.obj_func.up_down_limit[i][0], high=self.obj_func.up_down_limit[i][1], size=(1,))[0]
                        vetor_list[i] = round(new_vactor, 3)
                
                if not vetor_list in self.hmm_matrix:
                    break

            if self.obj_func.sample_size > len(self.obj_func.input_X):
                raise Exception("sample_size can't larger then input size.")
            
            error = self.obj_func.fitness(vetor_list)

            overwrite_index = hmm_err_list.index(max(hmm_err_list))
            if hmm_err_list[overwrite_index] > error:
                print('HMM_UPDATEP_NEW_VECTOR:',vetor_list)
                print('HMCR:',self.obj_func.hmcr_proba,'PAR:',self.obj_func.par_proba,'HMM_UPDATE_NEW_ERROR:',error)
                hmm_err_list[overwrite_index] = error
                self.hmm_matrix[overwrite_index] = vetor_list

        return self.hmm_matrix,hmm_err_list,hmm_err_list.index(min(hmm_err_list))