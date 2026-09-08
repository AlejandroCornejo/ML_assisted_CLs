"""Second coupon experiment: independent volumetric response and learned features.

An affine-in-J reference correction preserves convexity in (F,J) for either
sign of reference pressure. A separate positive logarithmic barrier restores
growth at J->0. This avoids tying volumetric stiffness to the feature slopes.
Global nonnegative energy is a separate question; see the run's audits.
"""
import numpy as np
import torch
from torch import nn

from enriched_pann import EnrichedEnergy, inverse_positive, paired_design, PositiveSplineLayer


class IntegratedHatSplineLayer(PositiveSplineLayer):
    """C2 convex monotone splines with arbitrary nonnegative linear curvature.

    The second derivative of each basis is a compact triangular hat, so
    curvature may both increase and decrease while staying nonnegative.
    Both tails are analytically linear with exactly matching derivatives.
    """
    def forward(self,x):
        h=self.knots[1]-self.knots[0]
        s=(x[:,:,None]-self.knots)/h
        t=s.clamp(-1.,1.)
        inside=torch.where(t<=0,(t+1).pow(3)/6,(-t.pow(3)+3*t.square()+3*t+1)/6)
        basis=h.square()*torch.where(s<=-1,torch.zeros_like(s),torch.where(s>=1,s,inside))
        return (x@torch.nn.functional.softplus(self.raw_linear).T
                +torch.einsum('nik,oik->no',basis,torch.nn.functional.softplus(self.raw_cubic))+self.bias)


def flexible_dictionary():
    rows=[]
    for angle in np.arange(12)*np.pi/12:
        for p in (.5,1.,1.5,2.,3.):
            for q in (.5,1.,1.5,2.,3.):
                for b in sorted(set((0.,p-.5,2*p-1))):
                    for c in sorted(set((0.,q-.5,2*q-1))):
                        rows.append((angle,p,q,b,c))
    return np.asarray(rows)


def affine_design(e,specs):
    a,b=paired_design(e,specs)
    j=np.sqrt((1+2*e[:,0])*(1+2*e[:,1])-e[:,2]**2)
    gj=np.column_stack((1+2*e[:,1],1+2*e[:,0],-e[:,2]))/j[:,None]
    rho=np.r_[2-specs[:,3]/specs[:,1]-specs[:,4]/specs[:,2],0.]
    a+=(np.log(j)-j+1)[:,None]*rho
    b+=(gj*(1/j-1)[:,None])[:,:,None]*rho
    return (np.column_stack((a,j-1-np.log(j))),
            np.concatenate((b,(gj*(1-1/j)[:,None])[:,:,None]),axis=2))


class FlexibleEnergy(EnrichedEnergy):
    def __init__(self,strain_scale,specs,*,core='icnn',widths=(24,24),seed=0,
                 learn_features=False,dynamic_center=False,spline_basis='cubic_hinge',analytic_stress=False):
        table=torch.as_tensor(specs,dtype=torch.float64)
        if table.ndim!=2 or table.shape[1]!=5 or not torch.isfinite(table).all():
            raise ValueError('Expected finite n by 5 feature specifications.')
        _,p,q,b,c=table.T
        if torch.any((p<.5)|(q<.5)|(b<0)|(c<0)|(b>2*p-1)|(c>2*q-1)):
            raise ValueError('Features are not convex in independent (F,J).')
        # Parent's log-pressure rho condition is unnecessary with affine J.
        safe=table.clone();safe[:,3:]=0
        super().__init__(strain_scale,safe.numpy(),core=core,widths=widths,seed=seed)
        self.specs.copy_(table)
        self.learn_features=learn_features
        self.dynamic_center=dynamic_center
        self.spline_basis=spline_basis
        self.analytic_stress=analytic_stress
        if spline_basis not in ('cubic_hinge','integrated_hat'):
            raise ValueError('Unknown spline basis')
        if core=='ickan' and spline_basis=='integrated_hat':
            dims=(len(table),*widths,1)
            self.base_icnn.layers=nn.ModuleList(IntegratedHatSplineLayer(a,b) for a,b in zip(dims[:-1],dims[1:]))
        self.raw_barrier=nn.Parameter(inverse_positive(torch.tensor(1.,dtype=torch.float64)))
        if learn_features:
            self.angles=nn.Parameter(table[:,0].clone())
            self.raw_powers=nn.Parameter(inverse_positive((table[:,1:3]-.5).clamp_min(1e-4)))
            fractions=(table[:,3:5]/(2*table[:,1:3]-1).clamp_min(1e-8)).clamp(.001,.999)
            self.raw_ratios=nn.Parameter(torch.logit(fractions))

    def effective_specs(self):
        if not self.learn_features: return self.specs
        pq=.5+torch.nn.functional.softplus(self.raw_powers)
        bc=(2*pq-1)*torch.sigmoid(self.raw_ratios)
        return torch.cat((self.angles[:,None],pq,bc),dim=1)

    def raw_features(self,x):
        c,_,j=self._kinematics(x)
        angle,p,q,b,d=self.effective_specs().T
        co,si=torch.cos(angle),torch.sin(angle)
        t=c[:,0,0,None]*co.square()+c[:,1,1,None]*si.square()+2*c[:,0,1,None]*co*si
        u=c[:,0,0,None]*si.square()+c[:,1,1,None]*co.square()-2*c[:,0,1,None]*co*si
        z=t.pow(p)*j[:,None].pow(-b)/p+u.pow(q)*j[:,None].pow(-d)/q
        return z,j

    def structural_features(self,x):
        z,j=self.raw_features(x)
        if self.dynamic_center:
            pq=self.effective_specs()[:,1:3]
            return z-(1/pq).sum(dim=1),j
        return z-self.feature_center,j

    def _reference_terms(self):
        if not self.analytic_stress: return super()._reference_terms()
        _,p,q,b,c=self.effective_specs().T
        center=1/p+1/q
        ref=(torch.zeros_like(center) if self.dynamic_center else center-self.feature_center)/self.feature_scale
        ref=ref[None,:]
        if not ref.requires_grad: ref=ref.requires_grad_(True)
        h0=self.base_icnn(ref)
        slope=torch.autograd.grad(h0.sum(),ref,create_graph=True)[0][0]/self.feature_scale
        return h0,(slope*(2-b/p-c/q)).sum()

    @property
    def barrier_coefficient(self):
        return torch.nn.functional.softplus(self.raw_barrier)+self.volumetric_floor

    def energy(self,x):
        features,j=self.structural_features(x)
        h=self.base_icnn(features/self.feature_scale)
        h0,r=self._reference_terms()
        c,_,_=self._kinematics(x)
        return (h-h0-r*(j-1)[:,None]
                +(self.barrier_coefficient*(j-1-torch.log(j))+.5*self.quadratic_coefficient*(j-1).square()
                  +self.volumetric_floor*(.5*(c[:,0,0]+c[:,1,1])-1-torch.log(j)))[:,None])

    def energy_and_stress(self,x,*,create_graph):
        if not self.analytic_stress: return super().energy_and_stress(x,create_graph=create_graph)
        if not x.requires_grad: x=x.requires_grad_(True)
        c,_,j=self._kinematics(x)
        angle,p,q,b,d=self.effective_specs().T
        co,si=torch.cos(angle),torch.sin(angle)
        dt=torch.stack((2*co.square(),2*si.square(),2*co*si),dim=0)
        du=torch.stack((2*si.square(),2*co.square(),-2*co*si),dim=0)
        t=1+(x*self.strain_scale)@dt;u=1+(x*self.strain_scale)@du
        aa=t.pow(p)*j[:,None].pow(-b);bb=u.pow(q)*j[:,None].pow(-d)
        raw=aa/p+bb/q
        center=1/p+1/q if self.dynamic_center else self.feature_center
        inputs=(raw-center)/self.feature_scale
        h=self.base_icnn(inputs)
        slope=torch.autograd.grad(h.sum(),inputs,create_graph=create_graph)[0]/self.feature_scale
        h0,r=self._reference_terms()
        gj=torch.stack((c[:,1,1],c[:,0,0],-c[:,0,1]),dim=1)/j[:,None]
        zjac=(aa/t)[:,None,:]*dt+(bb/u)[:,None,:]*du-((b*aa/p+d*bb/q)/j[:,None])[:,None,:]*gj[:,:,None]
        alpha,beta,eps=self.barrier_coefficient,self.quadratic_coefficient,self.volumetric_floor
        stress=(torch.einsum('ni,nji->nj',slope,zjac)
                +(-r+alpha*(1-1/j)+beta*(j-1))[:,None]*gj
                +eps*(torch.tensor([1.,1.,0.],dtype=x.dtype,device=x.device)-gj/j[:,None]))*self.strain_scale
        energy=h-h0+(-r*(j-1)+alpha*(j-1-torch.log(j))+.5*beta*(j-1).square()
                     +eps*(.5*(c[:,0,0]+c[:,1,1])-1-torch.log(j)))[:,None]
        return energy,stress

    def initialize(self,coefficients,style='nonlinear'):
        super().initialize_linear(coefficients[:-1])
        with torch.no_grad():
            self.raw_barrier.copy_(inverse_positive(torch.tensor(coefficients[-1],dtype=torch.float64).clamp_min(1e-6)))
            self.raw_quadratic.copy_(inverse_positive(torch.tensor(coefficients[-2],dtype=torch.float64).clamp_min(.01)))
            if style=='linear': return
            n=len(self.specs)
            if self.core_kind=='icnn':
                for weights,bias in zip(self.base_icnn.raw_input_weights,self.base_icnn.biases):
                    weights.copy_(inverse_positive(.5/np.sqrt(n)*torch.exp(.8*torch.randn_like(weights))))
                    bias.copy_(1.5*torch.randn_like(bias))
                for weights in self.base_icnn.raw_hidden_weights:
                    weights.copy_(inverse_positive(.2/np.sqrt(weights.shape[1])*torch.exp(.5*torch.randn_like(weights))))
                self.base_icnn.raw_output_hidden.fill_(-2.5)
                # Keep the fitted linear skip intact; curvature gets its own
                # independently calibrated amplitude in the trainer.
            else:
                for layer in self.base_icnn.layers:
                    ni=layer.raw_linear.shape[1]
                    layer.raw_linear.copy_(inverse_positive(.5/np.sqrt(ni)*torch.exp(.5*torch.randn_like(layer.raw_linear))))
                    layer.raw_cubic.copy_(inverse_positive(.05/np.sqrt(ni)*torch.exp(.5*torch.randn_like(layer.raw_cubic))))
                    layer.bias.copy_(.5*torch.randn_like(layer.bias))

    def certificate_summary(self):
        _,r=self._reference_terms()
        return dict(core=self.core_kind,learn_features=self.learn_features,
                    dynamic_center=self.dynamic_center,
                    spline_basis=self.spline_basis,
                    analytic_stress=self.analytic_stress,
                    specs=self.effective_specs().detach().cpu().tolist(),
                    normalization='affine J; reference pressure can have either sign',
                    reference_pressure=float(r.detach()),barrier=float(self.barrier_coefficient.detach()),
                    quadratic=float(self.quadratic_coefficient.detach()),
                    polyconvex_by_construction=True,
                    nonnegative_energy='not implied by normalization alone; requires separate audit')


def load_flexible(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False)
    model=FlexibleEnergy(**ck['configuration']).double().to(device)
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval()
    return model,ck


def physical_response(model,checkpoint,strain,*,tangent=False):
    """Evaluate physical W [Pa], second Piola S [Pa], and optional dS/dE [Pa].

    Input is an n by 3 array [E11,E22,gamma12], gamma12=2E12.
    This helper is an interface check, not an FE2 deployment adapter.
    """
    e=torch.as_tensor(strain,dtype=model.strain_scale.dtype,device=model.strain_scale.device)
    ss,es=checkpoint['strain_scale'],checkpoint['energy_scale']
    with torch.enable_grad():
        x=(e/ss).detach().requires_grad_(True)
        w,s=model.energy_and_stress(x,create_graph=tangent)
        result=dict(energy=w[:,0].detach().cpu().numpy()*es,
                    stress=s.detach().cpu().numpy()*es/ss)
        if tangent:
            d=torch.stack([torch.autograd.grad(s[:,j].sum(),x,retain_graph=True)[0] for j in range(3)],dim=1)
            result['tangent']=d.detach().cpu().numpy()*es/ss**2
    return result
