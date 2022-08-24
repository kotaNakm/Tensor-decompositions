from html import entities
from typing_extensions import Self
import numpy as np
import pandas as pd 
from importlib import import_module
import itertools

import sys
sys.path.append("_dat")
sys.path.append("_src")

import utils



class AGH(object):
    def __init__(
        self,
        entities,
        value_column,
        rank,
        initial_gamma,
        q,
        negative_curvature,
        optimization,
        phai,
        tensor_shape,
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

        self.target_vectors = np.zeros([[s, self.rank] for s in tensor_shape])

    def init_factors(self):
        factors=[]
        for s in self.tensor_shape:
            factors.append(np.rand((s,self.rank)))
        self.factors = factors
       

    def train(self,train_tensor):
        print(train_tensor)
        loss_logs=[]

        
        



    def optimazer():
        pass 


    def calc_loss(self,tensor):
        sum_error=0

        for entry in tensor:
            observed_val = entry[-1]
            
            sum_distance = self.calc_factors_distances()        
            estimated_val = self.projection_tanh(sum_distance)

            sum_error += (observed_val-estimated_val)**2

        return sum_error
        



    def calc_factors_distances(self,):
            target_vectors=[]
            for entity, factor in zip(entry, self.factors):
            # for entity, factor in zip(entry[:self.order],self.factors):

                factor
            # for combination
            # itertools
            #     sum_distance +=


    def projection_tanh(self,val):
        return self.phai*(1-np.tanh(val))

    def calc_distance(self, a, b):
        sqrt_c = np.sqrt(self.negative_curvature)
        mobius_addition = self.mobius_addition(-a,b)
        dist = 2/sqrt_c + np.arccosh(sqrt_c*mobius_addition)
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


    def reconstruction():
        pass 
