"""Analytic/autograd and structural-regression tests. Run with python3 -m unittest."""
import unittest

import numpy as np
import torch

from enriched_pann import (EnrichedEnergy, PositiveSplineKAN, candidate_specs,
                           paired_design, inverse_positive)
from anisotropic_pann_model import AnisotropicPolyconvexEnergy


class LinearCore(torch.nn.Module):
    def __init__(self,a):
        super().__init__()
        self.register_buffer('a',torch.as_tensor(a,dtype=torch.float64))

    def forward(self,x): return (x@self.a)[:,None]


def tangent(model,e):
    x=torch.tensor([e],dtype=torch.float64,requires_grad=True)/model.strain_scale
    _,s=model.energy_and_stress(x,create_graph=True)
    d=torch.stack([torch.autograd.grad(s[0,j],x,retain_graph=True)[0][0] for j in range(3)])
    return s.detach()/model.strain_scale,d.detach()/model.strain_scale**2


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.specs=np.array([[.2,1.,3.,0.,4.],[1.1,2.,3.,0.,0.]])

    def test_legacy_identities_even_with_more_rotations(self):
        models=[AnisotropicPolyconvexEnergy(strain_scale=.2),
                EnrichedEnergy(.2,include_original=True,n_rotations=12)]
        for model in models:
            s,d=tangent(model,[0.,0.,0.])
            self.assertLess(abs(float(d[0,0]-d[1,1])),1e-10)
            self.assertLess(abs(float(d[0,2]+d[1,2])),1e-10)
            for strain in (.03,.1):
                s,_=tangent(model,[strain,strain,0.])
                self.assertLess(abs(float(s[0,0]-s[0,1])),1e-10)
                self.assertLess(abs(float(s[0,2])),1e-10)

    def test_new_features_allow_missing_coupling(self):
        model=EnrichedEnergy(.2,self.specs)
        model.base_icnn=LinearCore([1.,.5])
        s,d=tangent(model,[0.,0.,0.])
        self.assertLess(float(s.abs().max()),1e-12)
        self.assertGreater(abs(float(d[0,0]-d[1,1])),.1)
        self.assertGreater(abs(float(d[0,2]+d[1,2])),.1)

    def test_analytic_design_matches_autograd(self):
        specs=candidate_specs()[::31]
        coefficients=np.random.default_rng(12).uniform(.01,.1,len(specs)+1)
        model=EnrichedEnergy(.2,specs)
        model.base_icnn=LinearCore(coefficients[:-1])
        with torch.no_grad(): model.raw_quadratic.copy_(inverse_positive(torch.tensor(coefficients[-1])))
        e=np.random.default_rng(13).uniform(-.05,.05,(17,3))
        a,b=paired_design(e,specs)
        w,s=model.energy_and_stress(torch.tensor(e/.2),create_graph=False)
        np.testing.assert_allclose(w.detach().numpy()[:,0],a@coefficients,atol=1e-10,rtol=1e-9)
        np.testing.assert_allclose(s.detach().numpy()/.2,b@coefficients,atol=1e-9,rtol=1e-9)

    def test_feature_convexity_independent_F_J(self):
        # This samples the actual certificate variables, not convexity in E.
        for row in candidate_specs()[::47]:
            angle,p,q,b,c=row
            d=torch.tensor([np.cos(angle),np.sin(angle)],dtype=torch.float64)
            perp=torch.tensor([-np.sin(angle),np.cos(angle)],dtype=torch.float64)
            def feature(y):
                f=y[:4].reshape(2,2);j=y[4]
                return (f@d).square().sum().pow(p)/p/j.pow(b)+(f@perp).square().sum().pow(q)/q/j.pow(c)
            for _ in range(3):
                y=torch.randn(5,dtype=torch.float64)*.2+torch.tensor([1.,0.,0.,1.,1.])
                h=torch.autograd.functional.hessian(feature,y)
                self.assertGreaterEqual(float(torch.linalg.eigvalsh(h).min()),-1e-8)

    def test_spline_convex_monotone_and_C2_knots(self):
        core=PositiveSplineKAN(2,(3,))
        for t in (-10.,-1.200001,-1.2,-1.199999,0.,1.199999,1.2,1.200001,5.):
            x=torch.tensor([t,.2],dtype=torch.float64,requires_grad=True)
            fun=lambda y: core(y[None,:])[0,0]
            grad=torch.autograd.functional.jacobian(fun,x)
            hess=torch.autograd.functional.hessian(fun,x)
            self.assertGreaterEqual(float(grad.min()),0.)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(hess).min()),-1e-10)
        layer=core.layers[0]
        for knot in layer.knots:
            hs=[]
            for side in (-1,1):
                y=torch.tensor([float(knot)+side*1e-7,.2],dtype=torch.float64)
                hs.append(torch.autograd.functional.hessian(lambda v: layer(v[None,:]).sum(),y))
            torch.testing.assert_close(hs[0],hs[1],atol=1e-7,rtol=1e-6)

    def test_reference_nonnegativity_consistency_and_domain(self):
        for core in ('icnn','ickan'):
            model=EnrichedEnergy(.2,self.specs,core=core,widths=(4,4))
            sample=torch.randn((40,3),dtype=torch.float64)*.2
            model.fit_scaling(sample)
            zero=torch.zeros((1,3),dtype=torch.float64)
            w,s=model.energy_and_stress(zero,create_graph=True)
            self.assertLess(float(w.abs().max()),1e-12)
            self.assertLess(float(s.abs().max()),1e-12)
            self.assertGreaterEqual(float(model.energy(sample).min()),-1e-12)
            xx=sample[:2].clone().requires_grad_(True)
            self.assertTrue(torch.autograd.gradcheck(model.energy,(xx,),eps=1e-6,atol=1e-5))
            self.assertTrue(torch.autograd.gradgradcheck(model.energy,(xx,),eps=1e-6,atol=1e-5))
            with self.assertRaises(ValueError): model.energy(torch.tensor([[-10.,-10.,0.]],dtype=torch.float64))

    def test_invalid_specs_rejected(self):
        for row in ([0,.4,1,0,0],[0,1,1,2,0],[0,3,3,5,5],
                    [0,3,3,3,3+1e-12],[0,float('nan'),1,0,0]):
            with self.assertRaises(ValueError): EnrichedEnergy(.2,[row])


if __name__=='__main__': unittest.main()
