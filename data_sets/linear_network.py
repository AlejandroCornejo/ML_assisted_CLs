import torch
import torch.nn as nn

class SymmetricLayer(nn.Module):
    def __init__(self, scale_estimate=1.,identity_init=False,size=3):
        super(SymmetricLayer, self).__init__()
        if identity_init:
            self.C = nn.Parameter(torch.eye(size,size))
        else:
            self.C = nn.Parameter(torch.abs(torch.randn(size,size)*scale_estimate))  # Random initialization

    def forward(self, x):
        return 0.5*torch.matmul(x,self.C+self.C.T)

class SimpleNet(nn.Module):
    def __init__(self,identity_init=False):
        super(SimpleNet, self).__init__()
        self.symmetric_layer = SymmetricLayer(identity_init=identity_init)

    def forward(self, x):
        return self.symmetric_layer(x)