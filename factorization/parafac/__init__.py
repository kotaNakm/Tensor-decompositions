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
        print(tensor_shape)

        # self.U= torch.nn.Parameter(torch.randn(rank, tensor_shape[0])/100., requires_grad=True)
        # self.V= torch.nn.Parameter(torch.randn(rank, tensor_shape[1])/100., requires_grad=True)
        # self.W= torch.nn.Parameter(torch.randn(rank, tensor_shape[2])/100., requires_grad=True)
         
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
        self.ein_str = src_str[:-1] + "->" + out_str
        print(self.ein_str)

    def forward(self):
        # reconstract tensor by factors
        # re_tensor = torch.zeros(*self.tensor_shape)
        # for ra in range(self.rank):
        #     UV = torch.outer(self.U[ra,:], self.V[ra,:])
        #     print(UV.shape)
        #     UV2 = UV.unsqueeze(2).repeat(1,1,self.tensor_shape[2]) # shape
        #     print(UV2.shape)
        #     W2 = self.W[ra,:].unsqueeze(0).unsqueeze(1).repeat(self.tensor_shape[0],self.tensor_shape[1],1)
        #     print(W2.shape)
        #     temp_tensor = UV2 * W2
        #     exit()
        #     re_tensor += temp_tensor
        re_tensor = torch.einsum(self.ein_str, *self.factors)
        return re_tensor

        # temp_matrix = torch.outer(self.factors[0][ra,:], self.factors[1][ra,:])     
        #  for mode in range(2, self.n_order):
        #     temp = torch.outer(temp_matrix, self.factors[mode][ra,:])

        # """
        # l x m x n
        # UV = torch.outer(u, v)
        # UV2 = UV.unsqueeze(2).repeat(1,1,n) # shape
        # W2 = w.unsqueeze(0).unsqueeze(1).repeat(l,m,1)
        # outputs = UV2 * W2  
        # """
        
    def loss(self, data, output, index): 
        mse = (torch.tensor(data,dtype=torch.float32) - output[tuple(index)])**2 # use tuple if you directly refer a elements avoiding fancy indexing

        # criterion = torch.nn.MSELoss()#reduction='sum'
        # mse = criterion(output[tuple(index)], torch.tensor(data,dtype=torch.float32))
        # print(output[tuple(index)].dtype)
        # print(torch.tensor(data).dtype)
        # mse.requires_grad = True
        
        return mse
