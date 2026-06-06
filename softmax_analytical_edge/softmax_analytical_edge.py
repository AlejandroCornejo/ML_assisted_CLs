import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torch.nn as nn
import torch.optim as optim

class SoftMaxAnalyticalEdge(nn.Module):
    """
    We will interpolate with 3 function candidates
        x
        x**2
        x**3

    f(x) = SUM PI_i * (c_i * funct(a_i * x + b_i) + d_i)

    PI is a softmax regularization
    We will train the parameters a_i, b_i, c_i, d_i and w_i (for PI)
    """
    def __init__(self):
        super().__init__()

        # self.a_i = nn.Parameter(torch.ones(3))
        # self.b_i = nn.Parameter(torch.zeros(3))
        # self.c_i = nn.Parameter(torch.ones(3))
        # self.d_i = nn.Parameter(torch.zeros(3))
        # multiples abcd induce cheating between experts

        self.a_i = nn.Parameter(torch.tensor(1.0))
        self.b_i = nn.Parameter(torch.tensor(0.0))
        self.c_i = nn.Parameter(torch.tensor(1.0))
        self.d_i = nn.Parameter(torch.tensor(0.0))

        # w_i = nn.Parameter(torch.ones(3)) # weights
        self.w_i = nn.Parameter(torch.zeros(3)) # weights

    def EvalFunctions(self, X):
        PI = torch.zeros(3)
        exp_w = torch.sum(torch.exp(self.w_i))
        PI = torch.exp(self.w_i) / exp_w

        return torch.vstack((
            # PI[0] * ((self.a_i[0]*X + self.b_i[0]) * self.c_i[0] + self.d_i[0]),
            # PI[1] * ((self.a_i[1]*X + self.b_i[1])**2 * self.c_i[1] + self.d_i[1]),
            # PI[2] * ((self.a_i[2]*X + self.b_i[2])**3 * self.c_i[2] + self.d_i[2])

            PI[0] * ((self.a_i*X + self.b_i)    * self.c_i + self.d_i),
            PI[1] * ((self.a_i*X + self.b_i)**2 * self.c_i + self.d_i),
            PI[2] * ((self.a_i*X + self.b_i)**3 * self.c_i + self.d_i)
        ))

    def forward(self, X):
        functs = self.EvalFunctions(X)
        return torch.sum(functs, axis=0)
