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
    def __init__(self, temperature=1.0, noisy_start=False):
        super().__init__()

        self.num_experts = 5

        if noisy_start:
            self.a_i = nn.Parameter(torch.tensor(np.random.uniform(0.5, 1.5, size=1), dtype=torch.float32))
            self.b_i = nn.Parameter(torch.tensor(np.random.uniform(-1.0, 1.0, size=1), dtype=torch.float32))
            self.c_i = nn.Parameter(torch.tensor(np.random.uniform(0.5, 1.5, size=1), dtype=torch.float32))
            self.d_i = nn.Parameter(torch.tensor(np.random.uniform(-1.0, 1.0, size=1), dtype=torch.float32))
            self.w_i = nn.Parameter(torch.tensor(np.random.uniform(0.1, 1.5, size=self.num_experts))) # weights
        else:
            self.a_i = nn.Parameter(torch.tensor(1.0))
            self.b_i = nn.Parameter(torch.tensor(0.0))
            self.c_i = nn.Parameter(torch.tensor(1.0))
            self.d_i = nn.Parameter(torch.tensor(0.0))
            self.w_i = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])) # weights

        self.temperature = temperature


    def GetExpertProbabilities(self):
        return nn.functional.softmax(self.w_i / self.temperature, dim=0) 

    def EvalFunctions(self, X):
        PI = self.GetExpertProbabilities()
        mod_X = self.a_i * X + self.b_i
        return torch.vstack((
            PI[0] * (mod_X   ),
            PI[1] * (mod_X**2),
            PI[2] * (mod_X**3),
            PI[3] * (torch.tanh(mod_X)),
            PI[4] * (torch.sin(mod_X))
        )) * self.c_i + self.d_i

    def forward(self, X):
        functs = self.EvalFunctions(X)
        return torch.sum(functs, axis=0)
