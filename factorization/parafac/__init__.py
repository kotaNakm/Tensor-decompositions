"""PARAFAC/CP/CANDECOMP implementation based on pytorch."""
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html


import numpy as np
import torch
import torch.nn.functional as F
import string

class PARAFAC(torch.nn.Module):
    """
    it supports arbitrary order tensors
    """

    def __init__(self, tensor_shape, rank):
        super().__init__()
        self.tensor_shape = tensor_shape
        self.rank = rank
        self.n_order = len(tensor_shape)
        
        print(f"Tensor shape:{tensor_shape}")

        self.factors = []
        for i, n_dim in enumerate(tensor_shape):
            temp = torch.nn.Parameter(torch.randn(rank, n_dim)/100., requires_grad=True)
            self.factors.append(temp)
        self.factors = torch.nn.ParameterList([f for f in self.factors])

        src_str=""
        out_str=""
        for mode in range(self.n_order):
            target_alphabet = string.ascii_lowercase[mode]
            src_str += "k" + target_alphabet + ","
            out_str += target_alphabet
        self.ein_str = src_str[:-1] + "->" + out_str #e.g., ak,bk,ck ->abc
        print(f"einsum:{self.ein_str}")

    def forward(self):
        re_tensor = torch.einsum(self.ein_str, *self.factors)
        return re_tensor

    def loss_record(self, data, output, index): 
        mse = (torch.tensor(data,dtype=torch.float32) - output[tuple(index)])**2 # use tuple if you directly refer a elements avoiding fancy indexing
        # criterion = torch.nn.MSELoss()#reduction='sum'
        # mse = criterion(output[tuple(index)], torch.tensor(data,dtype=torch.float32))        
        return mse

    def loss_tensor(self, tensor_data, output, loss_type="mse"):
        if loss_type=="mse": 
            mse = ((torch.tensor(tensor_data,dtype=torch.float32) - output)**2).sum() 
        return mse


    def training_tensor(self, tensor_data, n_iter, optimizer, loss_print_interval=1000,tol=1e-2):        
        for it in range(n_iter):
            optimizer.zero_grad()
            loss_out=0
            output = self.forward()
            loss_out = self.loss_tensor(tensor_data, output, loss_type="mse")
            loss_out.backward()
            optimizer.step()
            loss_val = loss_out.item()
            
            if it % loss_print_interval == 0:
                print(f"{it}: {loss_val}")
            if loss_val<tol:
                print(f"early stop at {it}")
                break
        
        return self.factors

    def training_record(self, record_data, n_iter, optimizer, loss_print_interval=10,tol=1e-2):
        indices = data[:,:-1].astype(int)
        values = data[:,-1]

        for it in range(n_iter):
            loss_out=0
            output = self.forward()
            for index, value in zip(indices, values):
                loss_out += self.loss_record(value, output, index)
            optimizer.zero_grad()
            loss_out.backward()
            optimizer.step()
            loss_val = loss_out.item()
            if it % loss_print_interval == 0:
                print(f"{it}: {loss_val}")
            if loss_val<tol:
                print(f"early stop at {it}")
                break

        return self.factors