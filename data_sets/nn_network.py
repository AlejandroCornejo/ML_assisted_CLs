import torch
import torch.nn as nn

class StrainInvariants(nn.Module):
    """Computes the first two invariants (I1, I2) from the Green-Lagrange strain tensor in Voigt form."""
    def forward(self, E_voigt):
        # E_voigt = E_voigt.requires_grad_(True)
        E11, E22, E12 = E_voigt[:,:, 0], E_voigt[:,:, 1], E_voigt[:,:, 2]

        # Compute first invariant I1 = Tr(E)
        I1 = E11 + E22

        # Compute second invariant I2 = det(E) in 2D, simplified
        I2 = E11 * E22 - (0.5 * E12) ** 2  # Voigt notation adjustment
        aux = torch.stack([I1, I2], dim=2)  # Shape: (batch_size, 2)
        return aux

class ConvexNN(nn.Module):
    """Convex neural network ensuring convexity in I1, I2."""
    def __init__(self, hidden_dim):
        super(ConvexNN, self).__init__()

        # Define unconstrained parameters
        scale = 1.0e-6
        self.raw_weight1 = nn.Parameter(torch.rand(hidden_dim, 2)*scale)
        self.raw_weight2 = nn.Parameter(torch.rand(hidden_dim, hidden_dim)*scale)
        self.raw_weight3 = nn.Parameter(torch.rand(1, hidden_dim)*scale)

        # NOTE AC: So this is a 2 x hidden_dim x 1    kind of ANN

    def forward(self, I):
        """Ensures convexity using non-negative weights and convex activations."""

        W1 = torch.relu(self.raw_weight1)
        W2 = torch.relu(self.raw_weight2)
        W3 = torch.relu(self.raw_weight3)

        # x = (I @ W1.T) ** 2  # Convex activation (square function)
        # x = (x @ W2.T) ** 2  # Convex activation
        # output = x @ W3.T    # Final convex output

        # NOTE: supone que I es un matriz de n_batch x 2  --> (I1, I2)
        x = I @ W1.T         # Convex activation (square function) //  CHECK: (n x n_hid) = (n x 2) @ (2 x n_hid)
        x = x @ W2.T         # Convex activation                   //  CHECK: (n x n_hid) = (n x n_hid) @ (n_hid x n_hid)
        output = x @ W3.T    # Final convex output                 //  CHECK: (n x 1) = (n x n_hid) @ (n_hid x 1) --> OK!!

        return output

class StrainEnergyPotential(nn.Module):
    """Computes the potential Psi(E) = ||E||^2 + convex function NN(I1, I2) and its derivative."""
    def __init__(self, hidden_dim=2,identity_init=True):
        if identity_init==False:
            raise Exception("expects the stress to be rescaled")
        super(StrainEnergyPotential, self).__init__()
        self.invariant_layer = StrainInvariants()
        self.convex_nn = ConvexNN(hidden_dim)

    def forward(self, E_voigt):
        """Compute strain energy potential Psi(E) and its derivative with respect to E_voigt."""
        # Compute invariants I1, I2
        I = self.invariant_layer(E_voigt)

        # Compute ||E||^2 (Frobenius norm squared)
        strain_norm_sq = torch.sum(E_voigt ** 2, dim=2, keepdim=True) # NOTE: Pregunta estupida, la norma es invariante? yo diría que no... Esto es un problema

        # Compute final potential
        #psi = 0.5*(strain_norm_sq + self.convex_nn(I))
        psi = 0.5*(self.convex_nn(I))

        # Compute the derivative of psi with respect to E_voigt using autograd
        # Since psi is (batch_size, 1), we sum it to get a scalar for grad computation
        d_psi_d_E = torch.autograd.grad(psi.sum(), E_voigt, create_graph=True)[0] # NOTE: Comentemos esto... que hace?

        #print(2.0*E_voigt - d_psi_d_E)
        #print(d_psi_d_E)

        #print(self.convex_nn.raw_weight1)

        return d_psi_d_E # NOTE: no deberia de hacer return psi ?????