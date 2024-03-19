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
        print(tensor_shape)

        self.U= torch.nn.Parameter(torch.randn(rank, tensor_shape[0])/100., requires_grad=True)
        self.V= torch.nn.Parameter(torch.randn(rank, tensor_shape[1])/100., requires_grad=True)
        self.W= torch.nn.Parameter(torch.randn(rank, tensor_shape[2])/100., requires_grad=True)

        # self.factors = []
        # for i, n_dim in enumerate(tensor_shape):
        #     temp = torch.nn.Parameter(torch.randn(rank, n_dim)/100., requires_grad=True)
        #     self.factors.append(temp)

    def forward(self, index):
        # reconstract tensor by factors
        print(self.tensor_shape)
        re_tensor = torch.zeros(*self.tensor_shape)
        for ra in range(self.rank):
            UV = torch.outer(self.U[ra,:], self.V[ra,:])
            print(UV.shape)
            UV2 = UV.unsqueeze(2).repeat(1,1,self.tensor_shape[2]) # shape
            print(UV2.shape)
            W2 = self.W[ra,:].unsqueeze(0).unsqueeze(1).repeat(self.tensor_shape[0],self.tensor_shape[1],1)
            print(W2.shape)
            temp_tensor = UV2 * W2
            print(temp_tensor.shape)
            re_tensor += temp_tensor.detach().clone()
        print(re_tensor.shape)
        print(index)
        # print(re_tensor[*[9,9,12]])
        output = re_tensor[index]
        return output

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
        
    def loss(self, data, output):
        mse = (data - output)**2
        # mse = np.sum((data - output)**2)
        return mse
