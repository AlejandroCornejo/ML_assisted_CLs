import numpy as np
import scipy as sp
import torch as torch

A = torch.tensor([[[1, 2, 3, 4],
                    [5, 6, 7, 8], 
                    [9, 10, 11, 12]],
                  [[13, 14, 15, 16], 
                  [17, 18, 19, 20], 
                  [21, 22, 23, 24]]])

# A can be seen as a 2 batch dataset of 4 entry vectors --> 2 batch x 3 steps x 4dim

print(torch.sum(A))
"""
tensor(300)
"""

print(torch.sum(A, axis=0)) # suma las matrices de cada batch
"""
tensor([[14, 16, 18, 20],
        [22, 24, 26, 28],
        [30, 32, 34, 36]])
"""



print(torch.sum(A, axis=2)) # para cada batch, colapsa las columnas
"""
tensor([[10, 26, 42],
        [58, 74, 90]])
"""