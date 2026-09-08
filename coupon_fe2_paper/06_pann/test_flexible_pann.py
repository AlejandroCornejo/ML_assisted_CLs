import unittest
import numpy as np
import torch

from flexible_pann import FlexibleEnergy, flexible_dictionary, affine_design,IntegratedHatSplineLayer,physical_response
from enriched_pann import inverse_positive
from train_flexible import tangent_dictionary


class Linear(torch.nn.Module):
    def __init__(self,coef):
        super().__init__();self.register_buffer('coef',torch.tensor(coef,dtype=torch.float64))
    def forward(self,x): return (x@self.coef)[:,None]


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(3)
        self.specs=flexible_dictionary()[::111]

    def test_analytic_affine_design_and_tangent(self):
        co=np.random.default_rng(4).uniform(.01,.1,len(self.specs)+2)
        model=FlexibleEnergy(.2,self.specs)
        model.base_icnn=Linear(co[:-2])
        with torch.no_grad():
            model.raw_quadratic.copy_(inverse_positive(torch.tensor(co[-2])))
            model.raw_barrier.copy_(inverse_positive(torch.tensor(co[-1])))
        e=np.random.default_rng(5).uniform(-.1,.1,(17,3))
        a,b=affine_design(e,self.specs)
        w,s=model.energy_and_stress(torch.tensor(e/.2),create_graph=False)
        np.testing.assert_allclose(w.detach().numpy()[:,0],a@co,atol=1e-9)
        np.testing.assert_allclose(s.detach().numpy()/.2,b@co,atol=1e-9)
        z=torch.zeros((1,3),dtype=torch.float64,requires_grad=True)
        w,s=model.energy_and_stress(z,create_graph=True)
        h=torch.stack([torch.autograd.grad(s[0,i],z,retain_graph=True)[0][0] for i in range(3)])/.2**2
        np.testing.assert_allclose(h.detach().numpy(),tangent_dictionary(self.specs)@co,atol=1e-8)

    def test_learned_constraints_reference_and_gradients(self):
        for core in ('icnn','ickan'):
            model=FlexibleEnergy(.2,self.specs,core=core,widths=(4,4),learn_features=True,dynamic_center=True,spline_basis='integrated_hat')
            x=torch.randn((3,3),dtype=torch.float64)*.1
            model.fit_scaling(x)
            with torch.no_grad():
                model.raw_powers.add_(torch.randn_like(model.raw_powers))
                model.raw_ratios.add_(torch.randn_like(model.raw_ratios))
                model.angles.add_(torch.randn_like(model.angles))
            _,p,q,b,c=model.effective_specs().T
            self.assertTrue(torch.all((p>=.5)&(q>=.5)&(b>=0)&(c>=0)&(b<=2*p-1)&(c<=2*q-1)))
            w,s=model.energy_and_stress(torch.zeros((1,3),dtype=torch.float64),create_graph=True)
            self.assertLess(float(w.abs().max()),1e-10);self.assertLess(float(s.abs().max()),1e-10)
            xx=x.clone().requires_grad_(True)
            self.assertTrue(torch.autograd.gradcheck(model.energy,(xx,),eps=1e-6,atol=1e-4))
            self.assertTrue(torch.autograd.gradgradcheck(model.energy,(xx,),eps=1e-6,atol=1e-4))
            wp,sp=model.energy_and_stress(xx,create_graph=True)
            (wp.square().sum()+sp.square().sum()).backward()
            for param in (model.raw_powers,model.raw_ratios,model.angles):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_integrated_hat_curvature_and_linear_tails(self):
        layer=IntegratedHatSplineLayer(1,1)
        x=torch.linspace(-3,3,1001,dtype=torch.float64)[:,None].requires_grad_(True)
        y=layer(x)
        dy=torch.autograd.grad(y.sum(),x,create_graph=True)[0]
        ddy=torch.autograd.grad(dy.sum(),x)[0]
        self.assertGreaterEqual(float(dy.min()),0.)
        self.assertGreaterEqual(float(ddy.min()),-1e-12)
        self.assertLess(float(ddy[[0,-1]].abs().max()),1e-12)
        self.assertGreater(float(ddy.max()),.001)
        for point in (-1.68,-1.2,0.,1.2,1.68):
            q=torch.tensor([[point]],dtype=torch.float64,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(layer,(q,),atol=1e-7))
            self.assertTrue(torch.autograd.gradgradcheck(layer,(q,),atol=1e-6))

    def test_analytic_chain_stress_and_parameter_gradients(self):
        for core in ('icnn','ickan'):
            model=FlexibleEnergy(.2,self.specs,core=core,widths=(4,4),learn_features=True,
                                 dynamic_center=True,spline_basis='integrated_hat',analytic_stress=True)
            xx=torch.randn((4,3),dtype=torch.float64)*.15
            model.fit_scaling(xx)
            x=xx.clone().requires_grad_(True)
            wa,sa=model.energy_and_stress(x,create_graph=True)
            w=model.energy(x)
            s=torch.autograd.grad(w.sum(),x,create_graph=True)[0]
            torch.testing.assert_close(wa,w,atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(sa,s,atol=1e-8,rtol=1e-8)
            pa=torch.autograd.grad(sa.square().sum()+wa.square().sum(),model.raw_powers,retain_graph=True)[0]
            pp=torch.autograd.grad(s.square().sum()+w.square().sum(),model.raw_powers)[0]
            torch.testing.assert_close(pa,pp,atol=1e-7,rtol=1e-7)

    def test_physical_units_and_batched_tangents(self):
        model=FlexibleEnergy(.2,self.specs,core='ickan',widths=(3,3),spline_basis='integrated_hat',analytic_stress=True)
        ck=dict(strain_scale=.2,energy_scale=2.7e7)
        e=np.array([[.02,-.01,.03],[.04,.01,-.02]])
        response=physical_response(model,ck,e,tangent=True)
        for j in range(3):
            delta=np.zeros_like(e);delta[:,j]=1e-6
            plus=physical_response(model,ck,e+delta)['stress']
            minus=physical_response(model,ck,e-delta)['stress']
            np.testing.assert_allclose((plus-minus)/2e-6,response['tangent'][:,:,j],rtol=2e-6,atol=.01)
        np.testing.assert_allclose(response['tangent'],response['tangent'].transpose(0,2,1),rtol=1e-12,atol=1e-7)


if __name__=='__main__': unittest.main()
