import torch
import torch.nn as nn

class StrainInvariants(nn.Module):
    """Computes the first two invariants (I1, I2) from the Green-Lagrange strain tensor in Voigt form."""
    def forward(self, tensor_voigt):
        # E_voigt = E_voigt.requires_grad_(True)
        E11, E22, E12 = tensor_voigt[:,:, 0], tensor_voigt[:,:, 1], tensor_voigt[:,:, 2]

        # Compute first invariant I1 = Tr(E)
        I1 = E11 + E22

        # Compute second invariant I2 = det(E) in 2D, simplified
        I2 = E11 * E22 - (0.5 * E12) ** 2  # Voigt notation adjustment
        aux = torch.stack([I1, I2], dim=2)  # Shape: (batch_size, 2)
        return aux

#ICNN as described in https://arxiv.org/pdf/1609.07152
class ConvexNN(nn.Module):
    """Convex neural network ensuring convexity in I1, I2."""
    def __init__(self, hidden_dim):
        super(ConvexNN, self).__init__()

        #self.W = nn.Parameter(torch.tensor([[1.0,0.0],[0.0,1.0]]))
        self.Wdiag = nn.Parameter(torch.tensor([1.0,1.0]))
        self.a_b = nn.Parameter(torch.tensor([1.0,1.0]))
        self.d_e = nn.Parameter(torch.tensor([-2.0,-1.0]))


        # Define unconstrained parameters
        # self.Wz1_nonneg = nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        # self.Wz2_nonneg = nn.Parameter(torch.rand(hidden_dim,1))

        # self.Wy0 = nn.Parameter(torch.rand(2,hidden_dim))
        # self.Wy1 = nn.Parameter(torch.rand(2,hidden_dim))
        # self.Wy2 = nn.Parameter(torch.rand(2,1))

        # self.b0 = nn.Parameter(torch.rand(1,hidden_dim))
        # self.b1 = nn.Parameter(torch.rand(1,hidden_dim))
        # self.b2 = nn.Parameter(torch.rand(1,1))

        # NOTE AC: So this is a 2 x hidden_dim x 1    kind of ANN

    def forward(self, input_vec):
        """
        input_vec: Tensor of shape (batch_size, 2)
        """

        log_input = torch.log(input_vec)

        # Linear transformation: W * [log(x), log(y)]^T + bias_exp
        #exp_inputs = torch.exp(log_input @ self.W.T) # + self.bias_exp)
        exp_inputs = torch.exp(log_input * self.Wdiag) # + self.bias_exp) #note the use of * instead of @ to keep W digaonal

        output = self.a_b * (exp_inputs + self.d_e)

        return output.sum(axis=2)  # Sum the contributions

        # Wz1 = torch.relu(self.Wz1_nonneg)
        # Wz2 = torch.relu(self.Wz2_nonneg)

        # z1 = torch.relu(I@self.Wy0 + self.b0)
        # z2 = torch.relu(z1@Wz1+I@self.Wy1  + self.b1)
        # z3 = torch.relu(z2@Wz2+I@self.Wy2  + self.b2)

        # return z3

class StrainEnergyPotential(nn.Module):
    def __init__(self, hidden_dim=2, identity_init=True):
        super(StrainEnergyPotential, self).__init__()
        self.invariant_layer = StrainInvariants()
        self.convex_nn = ConvexNN(hidden_dim)

    def EvaluatePsi(self, E_voigt):
        """Compute strain energy potential Psi(E)"""
        # Ensure we maintain gradient tracking
        identity = torch.tensor([1.0, 1.0, 0.0], device=E_voigt.device).view(1, 1, 3)
        C_voigt = E_voigt * 2.0 + identity
        I = self.invariant_layer(C_voigt)
        psi = 0.5 * (self.convex_nn(I))
        return psi

    def forward(self, E_voigt):
        # Ensure gradient tracking
        E_voigt = E_voigt.detach().requires_grad_(True)  # detach() avoids any existing graph

        psi = self.EvaluatePsi(E_voigt)

        # ALWAYS create graph - we need gradients for physics
        d_psi_d_E = torch.autograd.grad(
            outputs=psi,
            inputs=E_voigt,
            grad_outputs=torch.ones_like(psi),
            create_graph=True,  # Always True for physical gradients
            retain_graph=True,
            allow_unused=False
        )[0]

        assert not torch.allclose(d_psi_d_E, torch.zeros_like(d_psi_d_E)), "Gradients are zero!"

        # Zero strain computation
        zeros = torch.zeros_like(E_voigt).requires_grad_(True)
        psi_0 = self.EvaluatePsi(zeros)
        d_psi_d_E_0 = torch.autograd.grad(
            outputs=psi_0,
            inputs=zeros,
            grad_outputs=torch.ones_like(psi_0),
            create_graph=True,  # Also needed here
            retain_graph=True,
            allow_unused=False
        )[0]

        return d_psi_d_E - d_psi_d_E_0