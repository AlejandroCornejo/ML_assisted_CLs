"""Validation-led second-stage PANN search. Test/probe withheld by default."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import qr
from scipy.optimize import nnls

from flexible_pann import FlexibleEnergy, flexible_dictionary, affine_design, load_flexible
from enriched_pann import inverse_positive

HERE=Path(__file__).resolve().parent


def tangent_dictionary(specs):
    v=np.array([1.,1.,0.]);vv=np.outer(v,v);ll=np.diag([-2.,-2.,-1.])
    result=[]
    for angle,p,q,b,c in specs:
        co,si=np.cos(angle),np.sin(angle)
        dt=np.array([2*co*co,2*si*si,2*co*si]);du=np.array([2*si*si,2*co*co,-2*co*si])
        rho=2-b/p-c/q
        result.append((p-1)*np.outer(dt,dt)+(q-1)*np.outer(du,du)
                      -b*(np.outer(dt,v)+np.outer(v,dt))-c*(np.outer(du,v)+np.outer(v,du))
                      +(b*b/p+c*c/q-rho)*vv-2*ll)
    return np.stack(result+[vv,vv],axis=-1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--core',choices=['icnn','ickan'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=1800)
    ap.add_argument('--lbfgs',type=int,default=400)
    ap.add_argument('--seed',type=int,default=15)
    ap.add_argument('--features',type=int,default=32)
    ap.add_argument('--widths',default=None)
    ap.add_argument('--learn-features',action='store_true')
    ap.add_argument('--static-center',action='store_true')
    ap.add_argument('--spline-basis',choices=['cubic_hinge','integrated_hat'],default='integrated_hat')
    ap.add_argument('--init',choices=['linear','nonlinear','calibrated'],default='calibrated')
    ap.add_argument('--energy-weight',type=float,default=.2)
    ap.add_argument('--tangent-weight',type=float,default=0.)
    ap.add_argument('--lr',type=float,default=.005)
    ap.add_argument('--evaluate',action='store_true')
    ap.add_argument('--resume',type=Path,help='Continue a saved candidate in a new output directory.')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    data_path=HERE.parent/'03_data/data.npz';data=np.load(data_path)
    e,s,w=[data[k] for k in ('E_train','S_train','W_train')]
    order=np.random.default_rng(5).permutation(len(e));vi,fi=order[:742],order[742:]
    ss,es=float(abs(e).max()),float(abs(w).max())
    wn,sn=w/es,s*ss/es;wd,sd=np.mean(wn[fi]**2),np.mean(sn[fi]**2)
    C0=np.load(HERE.parent/'00_rve/C0_periodic.npz')['C0_periodic']/es
    specs=flexible_dictionary();a,b=affine_design(e[fi],specs)
    matrix=np.concatenate((a*np.sqrt(args.energy_weight/(wd*len(fi))),
                           (b*ss).reshape(-1,a.shape[1])/np.sqrt(sd*3*len(fi))))
    target=np.r_[wn[fi]*np.sqrt(args.energy_weight/(wd*len(fi))),sn[fi].ravel()/np.sqrt(sd*3*len(fi))]
    coef,_=nnls(matrix,target,maxiter=20000)
    ccoef,_=nnls(tangent_dictionary(specs).reshape(9,-1),C0.ravel(),maxiter=20000)
    selected=list(np.flatnonzero((coef[:-2]>1e-6)|(ccoef[:-2]>1e-6)))
    # QR adds independent response shapes; no validation or test labels enter.
    rows=np.linspace(0,len(matrix)-1,600,dtype=int)
    small=matrix[rows,:-2];small=small/np.linalg.norm(small,axis=0).clip(1e-10)
    _,_,pivot=qr(small,mode='economic',pivoting=True)
    for k in pivot:
        if len(selected)>=args.features: break
        if k not in selected: selected.append(int(k))
    specs=specs[selected];coef=np.r_[coef[selected],coef[-2:]]
    del matrix,a,b,small
    widths=tuple(map(int,args.widths.split(','))) if args.widths else ((24,24) if args.core=='icnn' else (8,8))
    config=dict(strain_scale=ss,specs=specs.tolist(),core=args.core,widths=widths,seed=args.seed,
                learn_features=args.learn_features,dynamic_center=not args.static_center,spline_basis=args.spline_basis,analytic_stress=True)
    model=FlexibleEnergy(**config).double()
    xt=torch.tensor(e/ss,dtype=torch.float64);st=torch.tensor(sn,dtype=torch.float64);wt=torch.tensor(wn,dtype=torch.float64)
    ct=torch.tensor(C0*ss**2,dtype=torch.float64)
    model.fit_scaling(xt[fi]);model.initialize(coef,args.init)
    if args.init=='calibrated':
        a,b=affine_design(e[fi[:500]],specs)
        _,initial=model.energy_and_stress(xt[fi[:500]].clone(),create_graph=False)
        nonlinear=initial.detach().numpy()-b@coef*ss
        factor=min(1.,.05*np.linalg.norm(sn[fi[:500]])/max(np.linalg.norm(nonlinear),1e-12))
        with torch.no_grad():
            params=[model.base_icnn.raw_output_hidden] if args.core=='icnn' else [
                model.base_icnn.layers[-1].raw_linear,model.base_icnn.layers[-1].raw_cubic]
            for param in params:
                param.copy_(inverse_positive(torch.nn.functional.softplus(param)*factor))
        print(json.dumps(dict(nonlinear_initial_amplitude_factor=factor)),flush=True)
    if args.resume:
        model,parent=load_flexible(args.resume)
        if parent['configuration']['core']!=args.core or parent['strain_scale']!=ss or parent['energy_scale']!=es:
            raise ValueError('Resume model core/normalization mismatch.')
        model.analytic_stress=True
        config=dict(parent['configuration'],analytic_stress=True)
    screen={}
    for name,ix in ([] if args.resume else [('fit',fi),('validation',vi)]):
        a,b=affine_design(e[ix],specs)
        screen[name]=dict(stress=float(np.linalg.norm(b@coef*ss-sn[ix])/np.linalg.norm(sn[ix])),
                          energy=float(np.linalg.norm(a@coef-wn[ix])/np.linalg.norm(wn[ix])))
    manifest=dict(configuration=config,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),screen=screen,
                  fit_count=len(fi),validation_count=len(vi),split_seed=5,
                  selection='resume' if args.resume else 'fit NNLS, reference tangent NNLS, then fit-only QR diversity',
                  test_probe_used=args.evaluate,linear_coefficients=None if args.resume else coef.tolist())
    if args.resume: manifest['resume_sha256']=hashlib.sha256(args.resume.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    np.savez(args.output/'split.npz',fit=fi,validation=vi)
    print(json.dumps(dict(core=args.core,features=len(model.specs),screen=screen)),flush=True)
    best=float('inf');state=None;beststep=0;history=[];t0=time.perf_counter()

    def losses(ix,graph):
        wp,sp=model.energy_and_stress(xt[ix].clone().requires_grad_(True),create_graph=graph)
        assert wp.shape==(len(ix),1)
        ls=(sp-st[ix]).square().mean()/sd;lw=(wp[:,0]-wt[ix]).square().mean()/wd
        return ls,lw

    def tangent_loss(graph):
        z=torch.zeros((1,3),dtype=torch.float64,requires_grad=True)
        _,s0=model.energy_and_stress(z,create_graph=True)
        cc=torch.stack([torch.autograd.grad(s0[0,k],z,retain_graph=True,create_graph=graph)[0][0] for k in range(3)])
        return (cc-ct).square().mean()/ct.square().mean(),cc

    def validate(step):
        nonlocal best,state,beststep
        ls,lw=losses(vi,False)
        score=float(ls.detach())
        if np.isfinite(score) and score<best:
            best=score;state=copy.deepcopy(model.state_dict());beststep=step
        row=dict(step=step,stress_rel=float(np.sqrt(score*sd/np.mean(sn[vi]**2))),
                 energy_rel=float(np.sqrt(float(lw.detach())*wd/np.mean(wn[vi]**2))),seconds=time.perf_counter()-t0)
        history.append(row)
        if step%100==0: print(json.dumps(row),flush=True)
        return score

    validate(0)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=40,factor=.5,min_lr=1e-5)
    def objective():
        ls,lw=losses(fi,True)
        total=ls+args.energy_weight*lw
        if args.tangent_weight: total=total+args.tangent_weight*tangent_loss(True)[0]
        if not torch.isfinite(total): raise RuntimeError('Nonfinite objective')
        return total
    for ep in range(1,args.epochs+1):
        optimizer.zero_grad(set_to_none=True);total=objective();total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),50.);optimizer.step()
        if ep%10==0: scheduler.step(validate(ep))
        if ep%200==0:
            torch.save(dict(configuration=config,state_dict=state,strain_scale=ss,energy_scale=es,best_step=beststep),args.output/'best.pt')
    model.load_state_dict(state)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=10,history_size=50,
                                tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad(set_to_none=True);total=objective();total.backward();return total
    for it in range(0,args.lbfgs,10):
        optimizer.step(closure);validate(args.epochs+it+10)
    model.load_state_dict(state)
    final={}
    for name,ix in [('fit',fi),('validation',vi)]:
        ls,lw=losses(ix,False)
        final[name]=dict(stress=float(np.sqrt(float(ls.detach())*sd/np.mean(sn[ix]**2))),
                        energy=float(np.sqrt(float(lw.detach())*wd/np.mean(wn[ix]**2))))
    if args.evaluate:
        for split in ('test','probe'):
            ee,ss0,ww=[data[f'{k}_{split}'] for k in ('E','S','W')]
            good=np.isfinite(ee).all(1)&np.isfinite(ss0).all(1)&np.isfinite(ww)
            wp,sp=model.energy_and_stress(torch.tensor(ee[good]/ss,dtype=torch.float64),create_graph=False)
            final[split]=dict(stress=float(np.linalg.norm(sp.detach().numpy()*es/ss-ss0[good])/np.linalg.norm(ss0[good])),
                              energy=float(np.linalg.norm(wp[:,0].detach().numpy()*es-ww[good])/np.linalg.norm(ww[good])),count=int(good.sum()))
    lc,cc=tangent_loss(False)
    wz,sz=model.energy_and_stress(torch.zeros((1,3),dtype=torch.float64),create_graph=False)
    results=dict(metrics=final,reference_tangent_relative_error=float(lc.detach().sqrt()),
                 reference_tangent=(cc.detach().numpy()*es/ss**2).tolist(),
                 reference_energy=float(wz.detach())*es,reference_stress=(sz.detach().numpy()*es/ss).tolist(),
                 best_step=beststep,seconds=time.perf_counter()-t0,certificate=model.certificate_summary())
    torch.save(dict(configuration=config,state_dict=state,strain_scale=ss,energy_scale=es,results=results),args.output/'model.pt')
    restored,_=load_flexible(args.output/'model.pt')
    torch.testing.assert_close(restored.energy(xt[vi[:4]]),model.energy(xt[vi[:4]]),rtol=0,atol=0)
    (args.output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__': main()
