function Kbdry = BoundaryWeightedGraphLaplacian(Kgraph,ny,nx,lambdaB,ellB)
%==========================================================================
% BoundaryWeightedGraphLaplacian
%
% PURPOSE
% -------
% Modify the graph Laplacian so that variations near the boundary of the
% latent grid are penalized more strongly.
%
% INPUTS
% ------
% Kgraph  : original Laplacian (Nm x Nm)
% ny, nx  : structured grid dimensions
% lambdaB : strength of boundary penalty (>=0)
% ellB    : decay length (>=1 recommended)
%
% OUTPUT
% ------
% Kbdry   : modified Laplacian (same size and structure)
%
% IDEA
% ----
% Each node gets a weight rho_j >= 1:
%   rho_j = 1 + lambdaB * exp(-d_j / ellB)
%
% where d_j = distance to boundary (in grid index sense).
%
% Each edge weight is scaled by:
%   rho_ab = (rho_a + rho_b)/2
%
%==========================================================================
% JAHO - MAW-ECM boundary regularization
%==========================================================================

Nm = ny*nx;

%--------------------------------------------------------------------------
% 1) Compute distance to boundary for each node
%--------------------------------------------------------------------------
D = zeros(ny,nx);

for iy = 1:ny
    for ix = 1:nx
        D(iy,ix) = min([iy-1, ny-iy, ix-1, nx-ix]);
    end
end

D = D(:);

%--------------------------------------------------------------------------
% 2) Nodal amplification factors
%--------------------------------------------------------------------------
rhoNode = 1 + lambdaB * exp(-D / max(ellB,1e-12));

%--------------------------------------------------------------------------
% 3) Extract edges from Kgraph
%--------------------------------------------------------------------------
[Iedge,Jedge,val] = find(triu(-Kgraph,1));  % positive edge weights

%--------------------------------------------------------------------------
% 4) Compute edge amplification
%--------------------------------------------------------------------------
rhoEdge = 0.5 * (rhoNode(Iedge) + rhoNode(Jedge));

valNew = val .* rhoEdge;

%--------------------------------------------------------------------------
% 5) Rebuild Laplacian
%--------------------------------------------------------------------------
Kbdry = sparse(Nm,Nm);

% off-diagonal
Kbdry = Kbdry + sparse(Iedge,Jedge,-valNew,Nm,Nm);
Kbdry = Kbdry + sparse(Jedge,Iedge,-valNew,Nm,Nm);

% diagonal
diagVals = -sum(Kbdry,2);
Kbdry = Kbdry + spdiags(diagVals,0,Nm,Nm);

end