"""PARAFAC/CP/CANDECOMP implementation based on pytorch."""
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html


import numpy as np
import torch
import torch.nn.functional as F



class PARAFAC(torch.nn.Module):
    """
    it only supports 3rd-order tensor
    """

    def __init__(self, tensor_shape, rank):

        super().__init__()
        self.tensor_shape = tensor_shape
        self.rank = rank
        self.n_order = len(tensor_shape) 
        self.factors = []

        for i, n_dim in enumerate(tensor_shape):
            temp = torch.nn.Parameter(torch.randn(rank, n_dim)/100., requires_grad=True)
            self.factors.append(temp)

    def forward(self, index):
        # reconstract tensor by factors
        print(self.tensor_shape)
        re_tensor = torch.zeros(*self.tensor_shape)
        for ra in range(self.rank):
            UV = torch.outer(self.factors[0][ra,:], self.factors[1][ra,:])
            print(UV.shape)
            UV2 = UV.unsqueeze(2).repeat(1,1,self.tensor_shape[2]) # shape
            print(UV2.shape)
            W2 = self.factors[2][ra,:].unsqueeze(0).unsqueeze(1).repeat(self.tensor_shape[0],self.tensor_shape[1],1)
            print(W2.shape)
            temp_tensor = UV2 * W2
            # temp_matrix = torch.outer(self.factors[0][ra,:], self.factors[1][ra,:])     
            # for mode in range(2, self.n_order):
            #     temp = torch.outer(temp_matrix, self.factors[mode][ra,:])
            re_tensor += temp_tensor.detach().clone()
        output = re_tensor[index]
        return output
        # """
        # l x m x n
        # UV = torch.outer(u, v)
        # UV2 = UV.unsqueeze(2).repeat(1,1,n) # shape
        # W2 = w.unsqueeze(0).unsqueeze(1).repeat(l,m,1)
        # outputs = UV2 * W2  
        # """
        

    def loss(self, data, output):
        mse = np.sum((data - output)**2)
        return mse

    def train(self, data, n_iter):
        indices = data[:,:-1]
        values = data[:,-1]

        for index, value, it in zip(indices, values, np.arange(n_iter)):
            output = self.forward(index)
            loss_out = self.loss(value, output)
            optimizer.zero_grad()
            loss_out.backward()
            optimizer.step()
        return self.factors, loss_logs