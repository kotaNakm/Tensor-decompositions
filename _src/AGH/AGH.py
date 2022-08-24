from html import entities
from typing_extensions import Self
import numpy as np

import pandas as pd 
from importlib import import_module
import itertools
from tqdm import tqdm

import sys
sys.path.append("_src")
import utils


class AGH(object):
    def __init__(
        self,
        tensor_shape,
        rank,
        initial_gamma,
        q,
        negative_curvature,
        optimization,
        phai,
        ):

        # data
        self.tensor_shape = tensor_shape
        self.order = len(tensor_shape)

        # model
        self.rank = rank
        self.initial_gamma = initial_gamma
        self.q = q
        self.negative_curvature = negative_curvature
        self.optimization = optimization
        self.phai = phai
        self.init_factors()

        self.target_vectors = [np.zeros((s, self.rank)) for s in tensor_shape]
        self.permutations = list(itertools.combinations([ i for i in range(self.order)],2))


    def init_factors(self):
        factors=[]
        for s in self.tensor_shape:
            factors.append(np.random.rand(s,self.rank))
        self.factors = factors
       

    def train(self,train_tensor):
        print(train_tensor)
        
        for target_factor_ind in range(self.order):
            self.optimizer(train_tensor,target_factor_ind)

    def optimizer(self,tensor, target_factor_ind):
        loss_logs=[]
        optimized_factor = self.factors[target_factor_ind]
        for iter in range(10):
            for entry in tqdm(tensor):
                optimized_entity = entry[target_factor_ind]
                # setup gamma 
                n_vec = np.sqrt(np.inner(optimized_factor[optimized_entity,:],optimized_factor[optimized_entity,:]))
                gamma = self.initial_gamma*np.exp(-n_vec)
                
                gradient = self.calc_gradient(entry)
                print(f"gradient:{gradient}")
                self.factors[target_factor_ind][optimized_entity,:] = optimized_factor[optimized_entity,:] - gamma*gradient

                loss = self.calc_loss(tensor)    
                print(f"loss:{loss}")
                loss_logs.append(loss)
    
    def calc_gradient(self,entry):
        obseved_val = entry[-1]
        sum_distance= self.calc_factors_distances(entry)
        term1 = 2*(obseved_val - self.projection_tanh(sum_distance))
        term2 = (-self.phai)*(1-np.tanh(sum_distance)**2)
        term3 = self.calc_factors_distances(entry,differential=True)
        
        # return term1
        return term1*term2*term3


    def calc_loss(self,tensor):
        sum_error=0
        for entry in tqdm(tensor):
            observed_val = entry[-1]
            sum_distance = self.calc_factors_distances(entry)        
            estimated_val = self.projection_tanh(sum_distance)

            sum_error += (observed_val-estimated_val)**2
        print(f"sum_error:{sum_error}")
        return sum_error
        

    def calc_factors_distances(self,entry,differential=False):
            for order, (each_entity, factor) in enumerate(zip(entry, self.factors)):
            # for entity, factor in zip(entry[:self.order],self.factors):
                self.target_vectors[order] = factor[each_entity,:]
            sum_distance=0
            for perm in self.permutations:
                ind1, ind2= perm
                sum_distance += self.calc_distance(self.target_vectors[ind1],self.target_vectors[ind2],differential) 
            
            return sum_distance

    def projection_tanh(self,val):
        return self.phai*(1-np.tanh(val))

    def calc_distance(self, a, b, differential=False):
        sqrt_c = np.sqrt(self.negative_curvature)
        mobius_addition = self.mobius_addition(-a,b)
        n_mobius = np.sqrt(np.inner(mobius_addition,mobius_addition))
        if differential:
            dist = 2/sqrt_c / ((sqrt_c*(n_mobius+1))**2-1)
        else:            
            dist = 2/sqrt_c * np.arccosh(sqrt_c*(n_mobius+1))
        # dist = 2/sqrt_c + np.arccosh(sqrt_c*(n_mobius+))
        return dist


    def mobius_addition(self, a,b):
        inner_ab=np.inner(a,b)
        n_a = np.inner(a,a)
        n_b = np.inner(b,b)
        c = self.negative_curvature
        
        num = (1+2*c*inner_ab+c*n_b)*a + (1-c*n_a)*b
        den = 1+2*c*inner_ab+c**2*n_a*n_b
        
        return num / den # as vector 


    def ascent():        
        pass 

