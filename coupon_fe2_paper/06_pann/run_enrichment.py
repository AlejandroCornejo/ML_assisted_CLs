"""Reproducible, isolated paired-feature screen and nonlinear training experiment.

python3 run_enrichment.py --core icnn --output enrichment_results/icnn_seed5
python3 run_enrichment.py --core ickan --output enrichment_results/ickan_seed5

No original checkpoint is modified. Selection/scaling use only the fitting
subset; validation selects checkpoints; test/probe are evaluated only at end.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls

from enriched_pann import EnrichedEnergy, candidate_specs, paired_design, load_enriched

HERE=Path(__file__).resolve().parent


def relative(pred,true):
    return float(np.linalg.norm(pred-true)/max(np.linalg.norm(true),1e-30))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--core',choices=['icnn','ickan'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=HERE.parent/'03_data/data.npz')
    ap.add_argument('--epochs',type=int,default=1500)
    ap.add_argument('--lbfgs',type=int,default=300)
    ap.add_argument('--seed',type=int,default=5)
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data=np.load(args.data)
    e,s,w=[data[k] for k in ('E_train','S_train','W_train')]
    w=w.reshape(-1)
    order=np.random.default_rng(5).permutation(len(e))
    vi,fi=order[:int(.15*len(e))],order[int(.15*len(e)):]
    # Match the established dataset normalization, not label-dependent features.
    ss,es=float(abs(e).max()),float(abs(w).max())
    wn,sn=w/es,s*ss/es
    wden,sden=np.mean(wn[fi]**2),np.mean(sn[fi]**2)
    specs=candidate_specs()
    a,b=paired_design(e[fi],specs)
    matrix=np.concatenate((a/np.sqrt(wden*len(fi)),
                           (b*ss).reshape(-1,a.shape[1])/np.sqrt(sden*3*len(fi))))
    target=np.r_[wn[fi]/np.sqrt(wden*len(fi)),sn[fi].ravel()/np.sqrt(sden*3*len(fi))]
    coeff,_=nnls(matrix,target,maxiter=10000)
    selected=np.flatnonzero(coeff[:-1]>1e-6)
    specs=specs[selected]
    coef=np.r_[coeff[selected],coeff[-1]]
    del matrix,a,b,target
    config=dict(strain_scale=ss,specs=specs.tolist(),core=args.core,
                widths=(24,24) if args.core=='icnn' else (8,8),seed=args.seed)
    model=EnrichedEnergy(**config).double()
    x=torch.tensor(e/ss,dtype=torch.float64)
    st,wt=torch.tensor(sn,dtype=torch.float64),torch.tensor(wn,dtype=torch.float64)
    model.fit_scaling(x[fi])
    model.initialize_linear(coef)
    screen={}
    for name,idx in [('fit',fi),('validation',vi)]:
        a,b=paired_design(e[idx],specs)
        screen[name]=dict(stress=relative(b@coef*ss,sn[idx]),energy=relative(a@coef,wn[idx]))
    manifest=dict(configuration=config,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  split_seed=5,fit_count=len(fi),validation_count=len(vi),
                  strain_scale=ss,energy_scale=es,screen=screen,linear_coefficients=coef.tolist(),
                  loss='relative mean squared stress + relative mean squared energy, fitting denominators',
                  checkpoint_selection='lowest validation stress MSE',
                  spline_variant='positive C2 truncated-power cubic edges, not legacy B-splines')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    np.savez(args.output/'split.npz',fit=fi,validation=vi)
    print(json.dumps(dict(core=args.core,features=len(specs),screen=screen)),flush=True)
    best=float('inf');best_state=None;best_step=0;history=[];t0=time.perf_counter()

    def loss(idx,graph):
        wp,sp=model.energy_and_stress(x[idx].clone().requires_grad_(True),create_graph=graph)
        assert wp.shape==(len(idx),1) and sp.shape==st[idx].shape
        ls=((sp-st[idx])**2).mean()/sden
        lw=((wp[:,0]-wt[idx])**2).mean()/wden
        return ls,lw

    def validate(step):
        nonlocal best,best_state,best_step
        ls,lw=loss(vi,False)
        ls,lw=float(ls.detach()),float(lw.detach())
        if np.isfinite(ls+lw) and ls<best:
            best=ls;best_step=step;best_state=copy.deepcopy(model.state_dict())
        row=dict(step=step,val_stress_mse=ls,val_energy_mse=lw,seconds=time.perf_counter()-t0)
        history.append(row)
        if step%100==0: print(json.dumps(row),flush=True)
        return ls

    validate(0)
    opt=torch.optim.Adam(model.parameters(),lr=.005)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=150,factor=.5,min_lr=1e-5)
    for ep in range(1,args.epochs+1):
        opt.zero_grad(set_to_none=True)
        ls,lw=loss(fi,True)
        total=ls+lw
        if not torch.isfinite(total): raise RuntimeError(f'Nonfinite loss at {ep}')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),50.)
        opt.step()
        if ep%10==0: scheduler.step(validate(ep))
    model.load_state_dict(best_state)
    opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=10,history_size=50,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(set_to_none=True)
        ls,lw=loss(fi,True)
        total=ls+lw
        if not torch.isfinite(total): raise RuntimeError('Nonfinite LBFGS loss')
        total.backward()
        return total
    for it in range(0,args.lbfgs,10):
        opt.step(closure)
        validate(args.epochs+it+10)
    model.load_state_dict(best_state)
    metrics={}
    def evaluate(ee,ss_true,ww_true):
        finite=np.isfinite(ee).all(axis=1)&np.isfinite(ss_true).all(axis=1)
        if ww_true is not None: finite &= np.isfinite(ww_true.reshape(-1))
        used,excluded=int(finite.sum()),int((~finite).sum())
        ee,ss_true=ee[finite],ss_true[finite]
        if ww_true is not None: ww_true=ww_true[finite]
        wp,sp=model.energy_and_stress(torch.tensor(ee/ss,dtype=torch.float64),create_graph=False)
        sp=sp.detach().numpy()*es/ss
        out=dict(stress=relative(sp,ss_true),stress_components=[relative(sp[:,j],ss_true[:,j]) for j in range(3)],used=used,excluded_nonfinite=excluded)
        if ww_true is not None: out['energy']=relative(wp[:,0].detach().numpy()*es,ww_true.reshape(-1))
        return out
    metrics['fit']=evaluate(e[fi],s[fi],w[fi])
    metrics['validation']=evaluate(e[vi],s[vi],w[vi])
    for split in ('test','probe'):
        if f'E_{split}' in data:
            metrics[split]=evaluate(data[f'E_{split}'],data[f'S_{split}'],data[f'W_{split}'] if f'W_{split}' in data else None)
    z=torch.zeros((1,3),dtype=torch.float64,requires_grad=True)
    wz,sz=model.energy_and_stress(z,create_graph=True)
    tangent=torch.stack([torch.autograd.grad(sz[0,j],z,retain_graph=True)[0][0] for j in range(3)])*es/ss**2
    results=dict(metrics=metrics,best_step=best_step,seconds=time.perf_counter()-t0,
                 reference_energy=float(wz.detach())*es,reference_stress=(sz.detach().numpy()*es/ss).tolist(),
                 reference_tangent=tangent.detach().numpy().tolist(),certificate=model.certificate_summary())
    checkpoint=dict(configuration=config,state_dict=best_state,strain_scale=ss,energy_scale=es,results=results)
    torch.save(checkpoint,args.output/'model.pt')
    reloaded,_=load_enriched(args.output/'model.pt')
    torch.testing.assert_close(model.energy(x[vi[:4]]),reloaded.energy(x[vi[:4]]),rtol=0,atol=0)
    (args.output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__': main()
