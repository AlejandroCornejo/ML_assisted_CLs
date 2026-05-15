function K = ComputeK_scl(COOR,CN,TypeElement)  
%==========================================================================
% ComputeK_scl
%==========================================================================
% PURPOSE
% -------
% Construct the graph-based smoothing matrix associated with the structured
% latent/master manifold used in the MAW-ECM procedure.
%
% The routine assembles a finite-element-like stiffness matrix over the
% structured grid of latent coordinates qM. The resulting operator acts as
% a discrete graph Laplacian and is later used to regularize the adaptive
% weight fields w(qM), promoting smooth variations of the cubature weights
% across neighboring manifold states.
%
% INPUT
% -----
% qMASTER
%   Matrix of latent/master coordinates:
%
%       size(qMASTER) = [nlatent, ns]
%
%   In the present implementation, nlatent = 2.
%
% DATA_interp
%   Structure containing the manifold-grid information. In particular,
%   the routine assumes that the sampled states can be reshaped into a
%   structured Cartesian grid according to:
%
%       DATA_interp.coeffRESHAPE_rows2_cols1
%
%   whose entries define the number of rows and columns associated with
%   the ordering of qMASTER.
%
% OUTPUT
% ------
% K_scl
%   Sparse graph-Laplacian-like matrix associated with the structured
%   latent grid. The matrix penalizes nonsmooth spatial variations of
%   quantities defined over the manifold samples.
%
%   If ns denotes the number of sampled manifold states,
%
%       K_scl in R^{ns x ns}.
%
% MATHEMATICAL INTERPRETATION
% ---------------------------
% The latent coordinates qM are interpreted as nodes of a structured
% two-dimensional mesh in parameter space. Neighboring samples are connected
% through bilinear finite elements, and the resulting operator corresponds
% to the stiffness matrix of a scalar diffusion problem on the manifold:
%
%       \int_{\Omega_q} grad(w) · grad(v) d\Omega_q .
%
% Consequently, minimizing expressions of the form
%
%       w^T K_scl w
%
% penalizes oscillatory weight distributions over the latent manifold and
% promotes smooth adaptive-weight fields.
%
% IMPORTANT REMARK
% ----------------
% The correctness of the graph operator relies entirely on the consistency
% between:
%
%   - the ordering of qMASTER,
%   - the reshaping convention encoded in
%         DATA_interp.coeffRESHAPE_rows2_cols1,
%   - and the ordering used later for the adaptive weights.
%
% Any mismatch between these orderings produces an incorrect graph topology
% and therefore invalid smoothing.
%
% IMPLEMENTATION NOTES
% --------------------
% The graph is constructed through an auxiliary structured quadrilateral
% mesh built directly in latent space. The resulting operator is assembled
% using standard FE-like connectivity and scalar Laplacian contributions.
%
% MAIN DEPENDENCIES
% -----------------
% quadMeshFromXY
% BoundaryWeightedGraphLaplacian
%
% J.A. Hernandez Ortega (JAHO)
%==========================================================================
if nargin == 0
    load('tmp1.mat')
end

% Dimensions of the problem 
nnode = size(COOR,1); ndim = size(COOR,2); nelem = size(CN,1); nnodeE = size(CN,2) ;     

% Shape function routines (for calculating shape functions and derivatives)
TypeIntegrand = 'K';
%dbstop('26')
[weig,posgp,shapef,dershapef] = ComputeElementShapeFun(TypeElement,nnodeE,TypeIntegrand) ; 

% Assembly of matrix K 
% ----------------
K = sparse(nnode,nnode) ; 
for e = 1:nelem 
%     if e== 30
%         disp('')
%     end
    % Conductivity matrix of element "e"
    ConductM = eye(ndim,ndim); 
    % Coordinates of the nodes of element "e"
    CNloc = CN(e,:) ; 
    Xe = COOR(CNloc,:)' ;
   %dbstop('39')
    % Computation of elemental conductance matrix 
    Ke = ComputeKeMatrix_Scl(ConductM,weig,dershapef,Xe) ; 
    for a=1:nnodeE 
        for b= 1:nnodeE 
            A = CN(e,a) ; 
            B = CN(e,b) ; 
            K(A,B) = K(A,B) + Ke(a,b) ; 
        end
    end
end
    
    