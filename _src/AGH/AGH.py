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

import warnings
warnings.simplefilter('error') # for catch 

SEED=100
ZERO=1e-15

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
        n_iter,
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
        self.permutations = list(
            itertools.combinations([i for i in range(self.order)], 2)
        )
        self.loss_logs = []
        self.gradient_logs = []

    def init_factors(self):
        factors = []        
        for s in self.tensor_shape:
            # factor = np.zeros((s, self.rank))
            # for i in range(s):
            #     # factor[i] = np.random.normal(mean,np.sqrt(self.rank),self.rank)
            #     factor[i] = np.random.normal(mean, std, self.rank) 
            # print(factor)
            factor = np.random.randn(s,self.rank)
            factors.append(factor)
        self.factors = factors

    def train(self, train_tensor):
        print("Train Tensor:")
        print(train_tensor)
        for _ in range(self.n_iter):
            loss = self.calc_loss(train_tensor, tqdm_=True)
            print(f"n_iter:{_}, exact_loss:{loss}")
            samples = random.sample(range(len(train_tensor)), len(train_tensor))
            for i in tqdm(samples):
                entry = train_tensor[i]
                pre_entry_error = self.calc_entry_error(entry)
                for target_factor_ind in range(self.order):
                    # print(f"order:{target_factor_ind}")
                    # loss = self.optimizer(train_tensor, target_factor_ind,entry)
                    self.optimizer(train_tensor, target_factor_ind,entry,loss)
                    # print(f"DONE! Optimization of Factor U^{target_factor_ind+1}")
                post_entry_error = self.calc_entry_error(entry)
                loss -= pre_entry_error - post_entry_error
                self.loss_logs.append(loss)
                print(f"approximated_loss:{loss}")
        return self.factors, self.loss_logs

    def optimizer(self, tensor, target_factor_ind,entry,loss):
        optimized_entity = entry[target_factor_ind]    
        gradient_descent = self.update_factor_gradient(target_factor_ind,optimized_entity,loss,tensor,entry)   
        self.gradient_logs.append(gradient_descent)

    def update_factor_gradient(self,target_factor_ind,optimized_entity,loss,tensor,entry):
        gradient_descent=0
        
        if loss/len(tensor) > self.l0: 
            # gradient Descent  
            # setup gamma
            n_vec = np.sqrt(
                np.inner(
                    self.factors[target_factor_ind][optimized_entity, :],
                    self.factors[target_factor_ind][optimized_entity, :]
                )
            )
            gamma = self.initial_gamma * np.exp(-n_vec) 
            gamma = ZERO if gamma < ZERO else gamma
            gradient_descent = self.calc_descent(entry, tensor)
            # print(f"Gradient:{gradient_descent}")
            # update factor
            self.factors[target_factor_ind][optimized_entity, :] -= gamma * gradient_descent
        
        else:
            # gradient Ascent with upper-bound of the Lipschitz constant
            Lq_w = self.calc_ascent(loss, entry, tensor)
            print("[Jump out]")
            print(f"Ascent:{Lq_w}")
            exit()
            # update factor
            self.factors[target_factor_ind][optimized_entity, :] += Lq_w
        # print(self.factors[target_factor_ind][optimized_entity, :])
        
        return gradient_descent

    def clip_subtensor_for_entry(self,tensor,entry,target_factor_ind):
        # clip subtensor which contains optimized_entity
        contain_idxs, contain_modes = np.where(tensor == entry)
        contain_idxs_mode = contain_idxs[contain_modes == target_factor_ind]
        # approximate loss using maximum 50 samples
        cliped_tensor = tensor[contain_idxs_mode[:50]]

    def calc_descent(self, entry, tensor):
        obseved_val = entry[-1]
        sum_distance = self.calc_factors_distances(entry)
        term1 = 2 * (obseved_val - self.projection_tanh(sum_distance))
        term2 = (-self.phai) * (1 - np.tanh(sum_distance) ** 2)
        term3 = self.calc_factors_distances(entry, differential=1)

        return term1 * term2 * term3 

    def calc_ascent(self, loss, entry, tensor):
        q = 2
        hessian = self.calc_hessian(entry)
        return q * self.L * loss + q * (q - 1) * 1 * hessian 

    def calc_hessian(self, entry):
        c = self.negative_curvature
        diff_D = self.calc_factors_distances(entry, differential=1)
        sum_distance = self.calc_factors_distances(entry)
        tanh_D = np.tanh(sum_distance)

        observed_val = entry[-1]
        estimated_val = self.projection_tanh(sum_distance)
        error = (observed_val - estimated_val) ** 2

        mobius_addition_sum = self.calc_factors_distances(entry, differential=2)

        const = (-2) * self.phai
        term1 = self.phai * tanh_D**2 * (1 - tanh_D**2) * diff_D**2
        term2 = error * (-2) * tanh_D * diff_D**2
        term3 = (
            error
            * (1 - tanh_D**2)
            * (-np.sqrt(c) * mobius_addition_sum)
            * ((np.sqrt(c) * mobius_addition_sum) ** 2 - 1)
        )

        return const * (term1 + term2 + term3)

    def calc_loss(self, tensor, tqdm_=False):
        sum_error = 0
        if tqdm_:
            for entry in tqdm(tensor):
                sum_error += self.calc_entry_error(entry)
        else:
            for entry in tensor:
                sum_error += self.calc_entry_error(entry)

        return sum_error

    def calc_entry_error(self, entry):
        observed_val = entry[-1]
        sum_distance = self.calc_factors_distances(entry)
        estimated_val = self.projection_tanh(sum_distance)
        # print(f"observed_val:{observed_val}")
        # print(f"estimated_val:{estimated_val}")
        return (observed_val - estimated_val) ** 2

    def calc_factors_distances(self, entry, differential=0):
        for order, (each_entity, factor) in enumerate(zip(entry, self.factors)):
            # print(order)
            # print(each_entity, factor)
            # for entity, factor in zip(entry[:self.order],self.factors):
            self.target_vectors[order] = factor[each_entity, :]
        sum_distance = 0
        for perm in self.permutations:
            ind1, ind2 = perm
            sum_distance += self.calc_distance(
                self.target_vectors[ind1], self.target_vectors[ind2], differential
            )
        return sum_distance

    def projection_tanh(self, val):
        return self.phai * (1 - np.tanh(val))

    def calc_distance(self, a, b, differential=0):
        sqrt_c = np.sqrt(self.negative_curvature)
        mobius_addition = self.mobius_addition(-a, b)
        n_mobius = np.sqrt(np.inner(mobius_addition, mobius_addition))
        n_mobius = n_mobius if n_mobius > ZERO else ZERO
        if differential == 0:
            dist = 2 / sqrt_c * np.arccosh(sqrt_c * (n_mobius + 1))
            # dist = 2/sqrt_c * np.arccosh(sqrt_c*(n_mobius))
        elif differential == 1:
            dist = 2 / sqrt_c / ((sqrt_c * (n_mobius + 1)) ** 2 - 1)
            # dist = 2/sqrt_c / ((sqrt_c*(n_mobius))**2-1)
        else:
            dist = n_mobius

        return dist

    def mobius_addition(self, a, b):
        inner_ab = np.inner(a, b)
        n_a = np.inner(a, a)
        n_b = np.inner(b, b)
        c = self.negative_curvature
        
        num = (1 + 2 * c * inner_ab + c * n_b) * a + (1 - c * n_a) * b
        den = 1 + 2 * c * inner_ab + c**2 * n_a * n_b

        return num / den  # as vector
