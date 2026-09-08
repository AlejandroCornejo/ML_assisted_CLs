"""Independent energy, derivative, and optional held-out audits of selected fits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flexible_pann import load_flexible

HERE=Path(__file__).resolve().parent


def energy_lower_bound_certificate(model):
    """Sufficient global nonnegativity test for THIS saved parameter set.

    With k=1/p+1/q and eta=(2-b/p-c/q)/k, Hadamard + minimization
    over perpendicular stretches gives z >= k J^eta. A supporting plane
    for the convex core then bounds W below by a scalar g(J), g(1)=g'(1)=0.
    J^2 g'' = alpha+eps+beta J^2 + sum h*k*eta*(eta-1)*J^eta.
    Positive lower bounds on this quantity certify W>=0 for all det F>0.
    """
    x=torch.zeros((1,3),dtype=torch.float64)
    z,_=model.structural_features(x)
    z=(z/model.feature_scale).detach().requires_grad_(True)
    h=(torch.autograd.grad(model.base_icnn(z).sum(),z)[0][0]/model.feature_scale).detach().numpy()
    specs=model.effective_specs().detach().numpy()
    p,q,b,c=specs[:,1:].T;k=1/p+1/q;eta=(2-b/p-c/q)/k
    coefficients=h*k*eta*(eta-1)
    alpha=float(model.barrier_coefficient.detach())+float(model.volumetric_floor)
    beta=float(model.quadratic_coefficient.detach())
    neg=coefficients<0;positive=coefficients>0
    negative_sum=float(-coefficients[neg].sum())
    low=alpha+coefficients[eta<0].sum()-negative_sum
    high=beta+coefficients[eta>=1].sum()-negative_sum
    result=dict(small_J_margin=float(low),large_J_slope_margin=float(high),
                negative_curvature_sum=negative_sum,
                certified=bool(low>=0 and high>=0),method='two-region analytic sufficient bound',
                scope='saved weights; not every possible optimizer iterate')
    if result['certified']: return result
    # More conservative interval bounds, plus analytic infinite tails.
    lower,upper=1e-4,1e4
    if neg.any():
        en=eta[neg].max()
        upper_tail=(coefficients[eta>=1]*upper**(eta[eta>=1]-en)).sum()+beta*upper**(2-en)-negative_sum
    else: upper_tail=alpha
    lower_tail=alpha+(coefficients[eta<0]*lower**eta[eta<0]).sum()+(coefficients[neg]*lower**eta[neg]).sum()
    edges=np.geomspace(lower,upper,4001)
    left,right=edges[:-1,None],edges[1:,None]
    pos_min=np.minimum(left**eta[positive],right**eta[positive])@coefficients[positive]
    neg_min=right**eta[neg]@coefficients[neg]
    margin=alpha+beta*edges[:-1]**2+pos_min+neg_min
    result.update(method='finite-interval power bounds with analytic infinite tails',
                  lower_tail_margin=float(lower_tail),upper_tail_margin=float(upper_tail),
                  interval_min_margin=float(margin.min()),
                  certified=bool(lower_tail>=0 and upper_tail>=0 and margin.min()>=0))
    return result


def evaluate(model,ck,e,s,w):
    good=np.isfinite(e).all(1)&np.isfinite(s).all(1)&np.isfinite(w)
    ss,es=ck['strain_scale'],ck['energy_scale']
    wp,sp=model.energy_and_stress(torch.tensor(e[good]/ss,dtype=torch.float64),create_graph=False)
    wp=wp.detach().numpy()[:,0]*es;sp=sp.detach().numpy()*es/ss
    if not np.isfinite(wp).all() or not np.isfinite(sp).all(): raise RuntimeError('Nonfinite prediction')
    return dict(stress=float(np.linalg.norm(sp-s[good])/np.linalg.norm(s[good])),
                stress_components=[float(np.linalg.norm(sp[:,j]-s[good,j])/np.linalg.norm(s[good,j])) for j in range(3)],
                energy=float(np.linalg.norm(wp-w[good])/np.linalg.norm(w[good])),
                stress_sample_relative_percentiles=np.percentile(np.linalg.norm(sp-s[good],axis=1)/np.linalg.norm(s[good],axis=1).clip(1e-12),[50,95,99,100]).tolist(),
                count=int(good.sum()),excluded_nonfinite=int((~good).sum()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('models',type=Path,nargs='+')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--evaluate',action='store_true',help='Only after model selection: reveal test/probe.')
    args=ap.parse_args();torch.set_num_threads(2)
    data=np.load(HERE.parent/'03_data/data.npz')
    report={}
    for path in args.models:
        model,ck=load_flexible(path)
        ss,es=ck['strain_scale'],ck['energy_scale']
        record={'nonnegative_energy_certificate':energy_lower_bound_certificate(model)}
        # Broad admissible C cloud, independent of all stored test/probe labels.
        rng=np.random.default_rng(101)
        angle=rng.uniform(-np.pi,np.pi,6000)
        lam=np.exp(rng.uniform(np.log(.25),np.log(3.),(6000,2)))
        co,si=np.cos(angle),np.sin(angle)
        c11=lam[:,0]**2*co**2+lam[:,1]**2*si**2
        c22=lam[:,0]**2*si**2+lam[:,1]**2*co**2
        c12=(lam[:,0]**2-lam[:,1]**2)*co*si
        e=np.column_stack(((c11-1)/2,(c22-1)/2,c12))
        sample=model.energy(torch.tensor(e/ss,dtype=torch.float64)).detach().numpy()
        record['broad_energy_sample']=dict(count=len(e),stretch_range=[.25,3.],
            finite=bool(np.isfinite(sample).all()),minimum=float(sample.min())*es,negative_count=int((sample < -1e-8).sum()))
        # Derivative checks at small states and selected fitting strains.
        tiny=torch.tensor([[.02,-.01,.03],[.1,-.04,-.08]],dtype=torch.float64,requires_grad=True)/ss
        record['gradcheck']=torch.autograd.gradcheck(model.energy,(tiny,),atol=1e-5,rtol=1e-4)
        record['gradgradcheck']=torch.autograd.gradgradcheck(model.energy,(tiny,),atol=1e-5,rtol=1e-4)
        minimum=float('inf');symmetry=0.
        # Hessian in F, then only rank-one directions: not convexity in strain.
        for _ in range(24):
            f=torch.eye(2,dtype=torch.float64)+.12*torch.tensor(rng.normal(size=(2,2)),dtype=torch.float64)
            def energy_F(flat):
                ff=flat.reshape(2,2);c=ff.T@ff
                ee=torch.stack(((c[0,0]-1)/2,(c[1,1]-1)/2,c[0,1]))
                return model.energy(ee[None,:]/ss)[0,0]
            h=torch.autograd.functional.hessian(energy_F,f.flatten()).detach().numpy()*es
            symmetry=max(symmetry,float(abs(h-h.T).max()))
            aa=rng.normal(size=(100,2));bb=rng.normal(size=(100,2))
            aa/=np.linalg.norm(aa,axis=1)[:,None];bb/=np.linalg.norm(bb,axis=1)[:,None]
            rank=np.einsum('ni,nj->nij',aa,bb).reshape(-1,4)
            minimum=min(minimum,float(np.einsum('ni,ij,nj->n',rank,h,rank).min()))
        record['rank_one_sample']=dict(count=2400,minimum_Pa=minimum,hessian_max_asymmetry_Pa=symmetry)
        if args.evaluate:
            record['metrics']={split:evaluate(model,ck,*[data[f'{key}_{split}'] for key in ('E','S','W')]) for split in ('test','probe')}
            labels=data['labels_probe'].astype(str)
            rings=np.array([float(label.rsplit('_',1)[1][:-1]) for label in labels])
            record['probe_by_ring']={}
            for ring in sorted(set(rings)):
                mask=rings==ring
                record['probe_by_ring'][str(ring)]=evaluate(model,ck,*[data[f'{key}_probe'][mask] for key in ('E','S','W')])
        report[str(path)]=record
        print(json.dumps({str(path):record}),flush=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
