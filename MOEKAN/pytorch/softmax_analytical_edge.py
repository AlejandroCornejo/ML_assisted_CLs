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
        tanh(x)
        sin(x)

    f(x) = SUM PI_i * (c_i * funct(a_i * x + b_i) + d_i)

    PI is a softmax regularization
    We will train the parameters a_i, b_i, c_i, d_i and w_i (for PI)
    """
    def __init__(self, temperature=1.0):
        super().__init__()

        self.num_experts = 5

        self.a_i = nn.Parameter(torch.tensor(1.0))
        self.b_i = nn.Parameter(torch.tensor(0.0))
        self.c_i = nn.Parameter(torch.tensor(1.0))
        self.d_i = nn.Parameter(torch.tensor(0.0))

        self.temperature = temperature

        # self.w_i = nn.Parameter(torch.zeros(self.num_experts)) # weights
        self.w_i = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])) # weights

    def GetExpertProbabilities(self):
        exp_w = torch.sum(torch.exp(self.w_i / self.temperature))
        PI = torch.exp(self.w_i / self.temperature) / exp_w
        return PI

    def EvalFunctions(self, X):
        PI = self.GetExpertProbabilities()
        return torch.vstack((
            PI[0] * ((self.a_i*X + self.b_i)    * self.c_i + self.d_i),
            PI[1] * ((self.a_i*X + self.b_i)**2 * self.c_i + self.d_i),
            PI[2] * ((self.a_i*X + self.b_i)**3 * self.c_i + self.d_i),
            PI[3] * (torch.tanh(self.a_i*X + self.b_i) * self.c_i + self.d_i),
            PI[4] * (torch.sin(self.a_i*X + self.b_i) * self.c_i + self.d_i)
        ))

    def forward(self, X):
        functs = self.EvalFunctions(X)
        return torch.sum(functs, axis=0)
