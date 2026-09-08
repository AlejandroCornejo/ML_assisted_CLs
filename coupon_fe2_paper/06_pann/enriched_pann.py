"""Coupon-local polyconvex feature experiments; shared manuscript models stay intact.

A paired feature is

    z = |F d|^(2p) / (p J^b) + |F d_perp|^(2q) / (q J^c).

For p,q >= 1/2 and 0 <= b <= 2p-1, 0 <= c <= 2q-1 each
term is convex in the independent variables (F,J). Its reference derivative
in engineering Green strain is rho*[1,1,0], rho=2-b/p-c/q.
We require rho >= 0. A convex nondecreasing core therefore retains the
original nonnegative pressure and -pressure*log(J) reference correction.
The mixed exponents remove the old same-power bank's forced hydrostatic
response and its tangent identities D11=D22, D13+D23=0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[1] / 'RVE_NeoHookean_Homogenization' / 'pann' / 'anisotropic'
sys.path.insert(0, str(SHARED))
from anisotropic_pann_model import AnisotropicPolyconvexEnergy, PositiveICNN


def inverse_positive(value):
    """Stable inverse softplus, including large initialization coefficients."""
    return value + torch.log(-torch.expm1(-value))


class PositiveSplineLayer(nn.Module):
    """Globally convex nondecreasing C2 cubic-spline edges, with no tail splice.

    Each edge is b + softplus(a)*x + sum softplus(c_k)*(x-k)_+^3/6.
    This is a truncated-power spline basis, not the external B-spline code.
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        self.register_buffer('knots', torch.linspace(-1.2, 1.2, 6, dtype=torch.float64))
        self.raw_linear = nn.Parameter(torch.full((n_out,n_in), -3., dtype=torch.float64))
        self.raw_cubic = nn.Parameter(torch.full((n_out,n_in,6), -5., dtype=torch.float64))
        self.bias = nn.Parameter(.1*torch.randn(n_out,dtype=torch.float64))

    def forward(self,x):
        hinge = torch.relu(x[:,:,None]-self.knots).pow(3)/6
        return (x @ torch.nn.functional.softplus(self.raw_linear).T
                + torch.einsum('nik,oik->no',hinge,torch.nn.functional.softplus(self.raw_cubic))
                + self.bias)


class PositiveSplineKAN(nn.Module):
    """Input-convex KAN variant, including a positive direct input skip."""
    def __init__(self,n_inputs,widths):
        super().__init__()
        dims=(n_inputs,*widths,1)
        self.layers=nn.ModuleList(PositiveSplineLayer(a,b) for a,b in zip(dims[:-1],dims[1:]))
        self.raw_skip=nn.Parameter(torch.full((1,n_inputs),-4.,dtype=torch.float64))

    def forward(self,x):
        h=x
        for layer in self.layers: h=layer(h)
        return h+x@torch.nn.functional.softplus(self.raw_skip).T


class EnrichedEnergy(AnisotropicPolyconvexEnergy):
    def __init__(self, strain_scale, specs=(), *, core='icnn', widths=(24,24),
                 seed=0, n_rotations=0, include_original=False):
        if not np.isfinite(strain_scale) or strain_scale<=0:
            raise ValueError('strain_scale must be finite and positive.')
        if not isinstance(n_rotations,int) or n_rotations<0:
            raise ValueError('n_rotations must be a nonnegative integer.')
        if not widths or any(width<1 for width in widths):
            raise ValueError('Hidden widths must be positive.')
        super().__init__(strain_scale=strain_scale, widths=widths)
        self.core_kind=core
        self.include_original = include_original
        self.n_rotations = n_rotations
        if n_rotations:
            angles = torch.arange(n_rotations, dtype=torch.float64) * np.pi/n_rotations
            rot = torch.stack((torch.cos(angles), -torch.sin(angles),
                               torch.sin(angles), torch.cos(angles)),dim=-1).reshape(-1,2,2)
            self.directions = torch.einsum('aij,kdj->akdi',rot,self.directions).reshape(-1,3,2)
            self.direction_weights = self.direction_weights.repeat(n_rotations,1)
        table = torch.as_tensor(np.asarray(specs).reshape(-1,5),dtype=torch.float64)
        if not torch.isfinite(table).all():
            raise ValueError('Feature specifications must be finite.')
        if len(table):
            _,p,q,b,c=table.T
            if torch.any((p<.5)|(q<.5)|(b<0)|(c<0)|(b>2*p-1)|(c>2*q-1)|(b/p+c/q>2)):
                raise ValueError('Uncertified paired feature exponents.')
        self.register_buffer('specs',table)
        n = (1+4*len(self.directions)+2 if include_original else 0)+len(table)
        if n == 0: raise ValueError('No features.')
        self.feature_scale = torch.ones(n,dtype=torch.float64)
        self.register_buffer('feature_center',torch.zeros(n,dtype=torch.float64))
        if core=='icnn':
            self.base_icnn = PositiveICNN(n,widths)
        elif core=='ickan':
            self.base_icnn = PositiveSplineKAN(n,widths)
        elif core=='ickan_legacy':
            from anisotropic_pann_model_ickan_claude import ICKANCore
            self.base_icnn = ICKANCore(n,widths,grid=6,spline_order=3,
                                       grid_range=(-1.2,1.2),seed=seed)
        else: raise ValueError(core)

    def _kinematics(self,x):
        if not torch.isfinite(x).all():
            raise ValueError('Strain inputs must be finite.')
        c,cof,j=super()._kinematics(x)
        if torch.any(c[:,0,0]<=0):
            raise ValueError('C must be positive definite, not just det(C)>0.')
        return c,cof,j

    def raw_features(self,x):
        c,_,j=self._kinematics(x)
        blocks=[]
        if self.include_original:
            blocks.append(super().structural_features(x)[0])
        if len(self.specs):
            angle,p,q,b,d=self.specs.T
            co,si=torch.cos(angle),torch.sin(angle)
            t=c[:,0,0,None]*co.square()+c[:,1,1,None]*si.square()+2*c[:,0,1,None]*co*si
            u=c[:,0,0,None]*si.square()+c[:,1,1,None]*co.square()-2*c[:,0,1,None]*co*si
            blocks.append(t.pow(p)*j[:,None].pow(-b)/p+u.pow(q)*j[:,None].pow(-d)/q)
        return torch.cat(blocks,dim=1),j

    def structural_features(self,x):
        z,j=self.raw_features(x)
        return z-self.feature_center,j

    def fit_scaling(self,x):
        with torch.no_grad():
            z,_=self.raw_features(x)
            z0,_=self.raw_features(torch.zeros((1,3),dtype=x.dtype,device=x.device))
            self.feature_center.copy_(z0[0])
            self.feature_scale.copy_((z-z0).abs().amax(dim=0).clamp_min(1e-4))

    def energy(self,x):
        # Independent positive barrier covers rho=0 dictionaries as well.
        w=super().energy(x)
        c,_,j=self._kinematics(x)
        return w + self.volumetric_floor*(.5*(c[:,0,0]+c[:,1,1])-1-torch.log(j))[:,None]

    def certificate_summary(self):
        _,pressure=self._reference_terms()
        return dict(core=self.core_kind, specs=self.specs.detach().cpu().tolist(),
                    include_original=self.include_original,n_rotations=self.n_rotations,
                    pressure=float(pressure.detach()),barrier_floor=float(self.volumetric_floor),
                    quadratic=float(self.quadratic_coefficient.detach()),
                    scope='2D J>0; convexity in independent (F,J); not universality or FE2 convergence',
                    convex_core_certified=self.core_kind in ('icnn','ickan'),
                    legacy_spline_caveat=self.core_kind=='ickan_legacy')

    def initialize_linear(self,coefficients):
        """Warm start from training-only nonnegative least squares, then train all weights."""
        a=torch.as_tensor(coefficients[:-1],dtype=self.feature_scale.dtype)*self.feature_scale
        with torch.no_grad():
            if self.core_kind=='icnn':
                self.base_icnn.raw_output_input.copy_(inverse_positive(a.clamp_min(1e-10))[None,:])
                self.base_icnn.raw_output_hidden.fill_(-7.)
                for weights in self.base_icnn.raw_input_weights: weights.fill_(-1.5)
            elif self.core_kind=='ickan':
                self.base_icnn.raw_skip.copy_(inverse_positive(a.clamp_min(1e-10))[None,:])
                self.base_icnn.layers[-1].raw_linear.fill_(-7.)
                self.base_icnn.layers[-1].raw_cubic.fill_(-9.)
            self.raw_quadratic.copy_(inverse_positive(torch.as_tensor(coefficients[-1],dtype=a.dtype).clamp_min(1e-10)))


def load_enriched(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    model=EnrichedEnergy(**checkpoint['configuration']).double().to(device)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    return model,checkpoint


def candidate_specs():
    """Fixed dictionary, declared independently of validation/test/probe labels."""
    pairs=[]
    for p in (.5,1.,1.5,2.,3.):
        for q in (.5,1.,1.5,2.,3.):
            for b in sorted(set((0.,p-.5,2*p-1))):
                for d in sorted(set((0.,q-.5,2*q-1))):
                    if b/p+d/q <= 2:
                        pairs.append((p,q,b,d))
    return np.asarray([(a,p,q,b,d) for a in np.arange(12)*np.pi/12 for p,q,b,d in pairs])


def paired_design(e,specs):
    """Physical energy and stress of each separately reference-corrected feature."""
    angle,p,q,b,d=np.asarray(specs).T
    co,si=np.cos(angle),np.sin(angle)
    dt=np.stack((2*co**2,2*si**2,2*co*si),axis=0)
    du=np.stack((2*si**2,2*co**2,-2*co*si),axis=0)
    t=1+e@dt;u=1+e@du
    j=np.sqrt((1+2*e[:,0])*(1+2*e[:,1])-e[:,2]**2)
    gj=np.column_stack((1+2*e[:,1],1+2*e[:,0],-e[:,2]))/j[:,None]
    rho=2-b/p-d/q
    a=t**p*j[:,None]**(-b);v=u**q*j[:,None]**(-d)
    w=(a-1)/p+(v-1)/q-rho*np.log(j[:,None])
    s=(a/t)[:,None,:]*dt+(v/u)[:,None,:]*du-((b*a/p+d*v/q+rho)/j[:,None])[:,None,:]*gj[:,:,None]
    return (np.column_stack((w,.5*(j-1)**2)),
            np.concatenate((s,((j-1)[:,None]*gj)[:,:,None]),axis=-1))
