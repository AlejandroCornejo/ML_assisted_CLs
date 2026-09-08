"""Replay saved fits, finite-label evaluation, and the rotated-bank control.

Writes an audit JSON only; never changes checkpoints or production models.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls

from enriched_pann import EnrichedEnergy, load_enriched

HERE=Path(__file__).resolve().parent


def rel(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def design_old(e,ss,rotations):
    model=EnrichedEnergy(ss,include_original=True,n_rotations=rotations)
    x=torch.tensor(e/ss,dtype=torch.float64,requires_grad=True)
    z,j=model.raw_features(x)
    x0=torch.zeros((1,3),dtype=torch.float64,requires_grad=True)
    z0,_=model.raw_features(x0)
    rho=torch.stack([torch.autograd.grad(z0[0,k],x0,retain_graph=True)[0][0,:2].mean()/ss for k in range(z.shape[1])])
    w=z-z0-rho*torch.log(j[:,None])
    w=torch.cat((w,.5*(j-1).square()[:,None]),dim=1)
    s=torch.stack([torch.autograd.grad(w[:,k].sum(),x,retain_graph=True)[0] for k in range(w.shape[1])],dim=2)
    return w.detach().numpy(),s.detach().numpy()


def legacy_spline_counterexample():
    path=Path('/home/kratos/ICKANs/ickan/spline.py')
    if not path.exists(): return {'skipped':'external legacy source not available'}
    spec=importlib.util.spec_from_file_location('legacy_spline_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    grid=torch.linspace(-2.,2.,13,dtype=torch.float64)[None,:]
    coef=torch.ones((1,1,9),dtype=torch.float64)
    result={}
    for name,bound in [('left',float(grid[0,3])),('right',float(grid[0,-4]))]:
        x=torch.tensor([[bound-1e-8],[bound+1e-8]],dtype=torch.float64,requires_grad=True)
        y,_=module.coef2curve(x,grid,coef,3)
        grad=torch.autograd.grad(y.sum(),x)[0][:,0].detach().numpy()
        result[name]=dict(slope_below=float(grad[0]),slope_above=float(grad[1]),downward_jump=float(grad[0]-grad[1]))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directories',nargs='+',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--controls',action='store_true')
    ap.add_argument('--refresh-reports',action='store_true',help='Refresh only the generated results.json evaluation metrics; weights stay untouched.')
    args=ap.parse_args()
    torch.set_num_threads(2)
    data=np.load(HERE.parent/'03_data/data.npz')
    e,s,w=[data[k] for k in ('E_train','S_train','W_train')]
    order=np.random.default_rng(5).permutation(len(e))
    vi,fi=order[:int(.15*len(e))],order[int(.15*len(e)):]
    report={'models':{},'legacy_spline_counterexample':legacy_spline_counterexample()}
    for directory in args.directories:
        model,ck=load_enriched(directory/'model.pt')
        ss,es=ck['strain_scale'],ck['energy_scale']
        out={}
        for split in ('test','probe'):
            ee,st,wt=[data[f'{k}_{split}'] for k in ('E','S','W')]
            good=np.isfinite(ee).all(axis=1)&np.isfinite(st).all(axis=1)&np.isfinite(wt)
            wp,sp=model.energy_and_stress(torch.tensor(ee[good]/ss,dtype=torch.float64),create_graph=False)
            wp,sp=wp.detach().numpy()[:,0]*es,sp.detach().numpy()*es/ss
            if not np.isfinite(wp).all() or not np.isfinite(sp).all(): raise RuntimeError('Nonfinite prediction')
            out[split]=dict(stress=rel(sp,st[good]),energy=rel(wp,wt[good]),
                            stress_components=[rel(sp[:,j],st[good,j]) for j in range(3)],
                            used=int(good.sum()),excluded_nonfinite=int((~good).sum()))
        out['reference_tangent']=ck['results']['reference_tangent']
        report['models'][directory.name]=out
        if args.refresh_reports:
            path=directory/'results.json'
            old=json.loads(path.read_text())
            for split in ('test','probe'): old['metrics'][split]=out[split]
            old['evaluation_note']='Replayed with finite FOM label mask; checkpoint weights unchanged.'
            path.write_text(json.dumps(old,indent=2,allow_nan=False)+'\n')
    report['coarse_reference_tangent']=np.load(HERE.parent/'00_rve/C0_periodic.npz')['C0_periodic'].tolist()
    # Infer an independent reference tangent from nearby production-mesh data.
    near=np.linalg.norm(e,axis=1)<.04
    en=e[near]
    quadratic=np.column_stack((np.ones(len(en)),en,en[:,0]**2,en[:,1]**2,en[:,2]**2,
                               en[:,0]*en[:,1],en[:,0]*en[:,2],en[:,1]*en[:,2]))
    ls=np.linalg.lstsq(quadratic,s[near],rcond=None)[0]
    report['production_grid_local_tangent_estimate']=dict(count=int(near.sum()),radius=.04,
                                                        tangent=ls[1:4].T.tolist(),
                                                        note='quadratic local stress regression, not an exact tangent')
    if args.controls:
        ss,es=float(abs(e).max()),float(abs(w).max())
        wn,sn=w/es,s*ss/es
        wd,sd=np.mean(wn[fi]**2),np.mean(sn[fi]**2)
        report['same_power_linear_controls']={}
        for rotations in (0,12):
            a,b=design_old(e,ss,rotations)
            mat=np.concatenate((a[fi]/np.sqrt(wd*len(fi)),b[fi].reshape(-1,a.shape[1])/np.sqrt(sd*3*len(fi))))
            yy=np.r_[wn[fi]/np.sqrt(wd*len(fi)),sn[fi].ravel()/np.sqrt(sd*3*len(fi))]
            co,_=nnls(mat,yy,maxiter=10000)
            report['same_power_linear_controls'][str(rotations)]=dict(features=a.shape[1]-1,
                fit_stress=rel(b[fi]@co,sn[fi]),validation_stress=rel(b[vi]@co,sn[vi]),
                fit_energy=rel(a[fi]@co,wn[fi]),validation_energy=rel(a[vi]@co,wn[vi]))
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
