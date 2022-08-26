from html import entities
from syslog import LOG_SYSLOG
from typing_extensions import Self
import numpy as np
import random
import copy

import pandas as pd 
from importlib import import_module
import itertools
from tqdm import tqdm

import sys
sys.path.append("_src")
import utils

random.seed(100)
np.random.seed(100)

class AGH(object):
    def __init__(
        self,
        tensor_shape,
        rank,
        initial_gamma,
        l0,
        L,
        negative_curvature,
        optimization,
        phai,
        n_iter
        ):

        # data
        self.tensor_shape = tensor_shape
        self.order = len(tensor_shape)

        # model
        self.rank = rank
        self.initial_gamma = initial_gamma
        self.l0 = l0
        self.L = L
        self.negative_curvature = negative_curvature
        self.optimization = optimization
        self.phai = phai
        self.n_iter = n_iter
        self.init_factors()

        self.target_vectors = [np.zeros((s, self.rank)) for s in tensor_shape]
        self.permutations = list(itertools.combinations([ i for i in range(self.order)],2))
        self.loss_logs = []

    def init_factors(self):
        factors=[]
        for s in self.tensor_shape:
            # factors.append(np.random.rand(s,self.rank)/np.sqrt(self.rank))
            factors.append(np.random.rand(s,self.rank))

        self.factors = factors

    def train(self,train_tensor):
        print(train_tensor)
        for target_factor_ind in range(self.order):
            self.optimizer(train_tensor,target_factor_ind)
            print(f"DONE! Optimization of Factor U^{target_factor_ind+1}")
        return self.factors, self.loss_logs

    def optimizer(self,tensor, target_factor_ind):
        optimized_factor = self.factors[target_factor_ind]
        loss = self.calc_loss(tensor)
        print(f"initial_loss:{loss}")
        samples = random.sample(range(len(tensor)),len(tensor))        
        # samples = random.sample(range(len(tensor)),self.n_iter)        
        # for i in tqdm(samples):
        for _ in range(self.n_iter):
            for i in tqdm(samples):
                entry = tensor[i]
                # print(f"entry:{entry}")
                optimized_entity = entry[target_factor_ind]
                pre_entry_error = self.calc_entry_error(entry)

                # gradient Descent 
                if loss > self.l0:
                    # setup gamma 
                    n_vec = np.sqrt(np.inner(optimized_factor[optimized_entity,:],optimized_factor[optimized_entity,:]))
                    gamma = self.initial_gamma*np.exp(-n_vec)                
                    
                    gradient_descent = self.calc_descent(entry)
                    # print(f"Gradient:{gradient_descent}")
                    # update factor
                    self.factors[target_factor_ind][optimized_entity,:] = optimized_factor[optimized_entity,:] - gamma*gradient_descent
                
                # gradient Ascent 
                # with upper-bound of the Lipschitz constant
                else:
                    Lq_w = self.calc_ascent(loss,entry)
                    print("[Jump out]")
                    print(f"Ascent:{Lq_w}")
                    # update factor
                    self.factors[target_factor_ind][optimized_entity,:] = optimized_factor[optimized_entity,:] + Lq_w
                post_entry_error = self.calc_entry_error(entry)
                loss -= (pre_entry_error - post_entry_error)
                self.loss_logs.append(loss)
            print(f"n_iter:{_}, loss:{loss}")

    def calc_descent(self,entry):
        obseved_val = entry[-1]
        sum_distance= self.calc_factors_distances(entry)
        term1 = 2*(obseved_val - self.projection_tanh(sum_distance))
        term2 = (-self.phai)*(1-np.tanh(sum_distance)**2)
        term3 = self.calc_factors_distances(entry,differential=1)
        
        return term1*term2*term3

    def calc_ascent(self,loss,entry):
        q=2
        hessian = self.calc_hessian(entry)
        return q * self.L * loss + q * (q-1) * 1 * hessian
    
    def calc_hessian(self,entry):
        c = self.negative_curvature
        diff_D =  self.calc_factors_distances(entry,differential=1)
        sum_distance = self.calc_factors_distances(entry)
        tanh_D = np.tanh(sum_distance)

        observed_val=entry[-1];
        estimated_val=self.projection_tanh(sum_distance)
        error = (observed_val-estimated_val)**2

        mobius_addition_sum = self.calc_factors_distances(entry,differential=2)

        const = (-2)*self.phai
        term1 = self.phai*tanh_D**2*(1-tanh_D**2)*diff_D**2
        term2 = error*(-2)*tanh_D*diff_D**2
        term3 = error*(1-tanh_D**2) * (-np.sqrt(c)*mobius_addition_sum) * ((np.sqrt(c)*mobius_addition_sum)**2-1)

        return const*(term1+term2+term3)

    def calc_loss(self,tensor):
        sum_error=0
        for entry in tqdm(tensor):    
            sum_error += self.calc_entry_error(entry)
        return sum_error

    def calc_entry_error(self,entry):
        observed_val = entry[-1]
        sum_distance = self.calc_factors_distances(entry)        
        estimated_val = self.projection_tanh(sum_distance)
        return (observed_val-estimated_val)**2
        

    def calc_factors_distances(self,entry,differential=0):
        for order, (each_entity, factor) in enumerate(zip(entry, self.factors)):
        # for entity, factor in zip(entry[:self.order],self.factors):
            self.target_vectors[order] = factor[each_entity,:]
        sum_distance=0
        for perm in self.permutations:
            ind1, ind2 = perm
            sum_distance += self.calc_distance(self.target_vectors[ind1],self.target_vectors[ind2],differential) 
            
        return sum_distance

    # def update_target_vectors(self,entry):
        
    def projection_tanh(self,val):
        return self.phai*(1-np.tanh(val))

    def calc_distance(self, a, b, differential=0):
        sqrt_c = np.sqrt(self.negative_curvature)
        mobius_addition = self.mobius_addition(-a,b)
        n_mobius = np.sqrt(np.inner(mobius_addition,mobius_addition))
        if differential==0:
            dist = 2/sqrt_c * np.arccosh(sqrt_c*(n_mobius+1))
            # dist = 2/sqrt_c * np.arccosh(sqrt_c*(n_mobius))
        elif differential==1:
            dist = 2/sqrt_c / ((sqrt_c*(n_mobius+1))**2-1)
        else:
            dist = mobius_addition        

        return dist

    def mobius_addition(self, a,b):
        inner_ab=np.inner(a,b)
        n_a = np.inner(a,a)
        n_b = np.inner(b,b)
        c = self.negative_curvature
        
        num = (1+2*c*inner_ab+c*n_b)*a + (1-c*n_a)*b
        den = 1+2*c*inner_ab+c**2*n_a*n_b
        
        return num / den # as vector 