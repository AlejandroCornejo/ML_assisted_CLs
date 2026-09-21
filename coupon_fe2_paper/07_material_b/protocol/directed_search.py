"""Stronger post-test search in principal stretches; see accompanying rule."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import qmc
from protocol.audit_robustness import acoustic, sqrt_c, verification
from protocol.evaluate_final_models import verify_gate, load_model, predict, BASE, LOCK
from protocol.prepare_design import digest
from protocol.train_material_b import _atomic_json

OUTPUT = BASE/'results/directed_search'
RULE = Path(__file__).with_name('DIRECTED_SEARCH_INTERNAL.md')
LOGMIN, LOGSPAN = np.log(.55), np.log(1.5/.55)


def strain_from_unit(x):
    x = np.asarray(x).reshape(-1, 4)
    lam = np.exp(LOGMIN+LOGSPAN*x[:, :2])
    c, s = np.cos(np.pi*x[:, 2]), np.sin(np.pi*x[:, 2])
    l1, l2 = lam[:, 0]**2, lam[:, 1]**2
    return np.column_stack(((l1*c*c+l2*s*s-1)/2,
                            (l1*s*s+l2*c*c-1)/2, (l1-l2)*c*s))


def norm_e(x):
    e = strain_from_unit(x)[0]
    return float(e[0]**2+e[1]**2+.5*e[2]**2)


def witness(model, scales, x):
    e = strain_from_unit(x)
    _, s, d = predict(model, e, scales)
    f = sqrt_c(e)
    val, a, b = acoustic(f, s, d, np.pi*x[3])
    checks = verification(model, e, a[0], b, scales)
    return dict(curvature_Pa=float(val[0]), unit_parameters=np.asarray(x).tolist(),
                strain=e[0].tolist(), strain_tensor_norm=float(np.sqrt(norm_e(x))),
                F=f[0].tolist(), det_F=float(np.linalg.det(f[0])), a=a[0].tolist(), b=b.tolist(),
                verification=checks,
                negative_verified=bool(val[0]<-1e5 and checks['direct_autograd_Pa']<-1e5
                  and all(v<-1e5 and abs(v-val[0])/abs(val[0])<.01
                          for v in checks['centered_energy_differences_Pa'].values())))


def run_search(row, scales):
    model = load_model(row)
    xyz = qmc.Sobol(3, scramble=True, seed=20260919).random_base2(12)
    x = np.column_stack((xyz, np.zeros(len(xyz))))
    e = strain_from_unit(x)
    _, s, d = predict(model, e, scales)
    f = sqrt_c(e)
    v = np.full(len(e), np.inf)
    for k in range(64):
        values, _, _ = acoustic(f, s, d, k*np.pi/64)
        update = values < v
        v[update], x[update, 3] = values[update], k/64
    np.savez_compressed(OUTPUT/(row['slug']+'_cloud.npz'), strain=e,
                        unit_parameters=x, minimum_Pa=v)
    print(row['slug'], 'cloud min MPa', v.min()/1e6, 'negative states', (v<0).sum(), flush=True)
    candidates = [(float(v[i]), x[i].copy()) for i in range(len(v))]
    local_reports = []
    def objective(p):
        strain = strain_from_unit(p)
        _, st, dt = predict(model, strain, scales)
        value = float(acoustic(sqrt_c(strain), st, dt, np.pi*p[3])[0][0])
        candidates.append((value, np.array(p).copy()))
        return value/1e9
    for i in np.argsort(v)[:8]:
        opt = minimize(objective, x[i], method='Powell', bounds=[(0., 1.)]*4,
                       options=dict(maxfev=2000, xtol=1e-6, ftol=1e-8))
        local_reports.append(dict(success=bool(opt.success), message=str(opt.message),
                                  nfev=int(opt.nfev), value_Pa=float(opt.fun*1e9)))
    most_negative = min(candidates, key=lambda pair: pair[0])
    nearest_reports = []
    negative_ids = np.flatnonzero(v <= -1e6)
    seeds = sorted(negative_ids, key=lambda i: norm_e(x[i]))[:4]
    starts = [x[i] for i in seeds]
    starts += [p for value, p in sorted(candidates, key=lambda pair: pair[0])[:4]
               if value <= -1e6]
    for p in starts:
        opt = minimize(norm_e, p, method='SLSQP', bounds=[(0., 1.)]*4,
                       constraints=[dict(type='ineq', fun=lambda a: -objective(a)-.001)],
                       options=dict(maxiter=300, ftol=1e-11))
        value = objective(opt.x)*1e9
        nearest_reports.append(dict(success=bool(opt.success), message=str(opt.message),
                                    nfev=int(opt.nfev), nit=int(opt.nit),
                                    curvature_Pa=value, norm=float(np.sqrt(norm_e(opt.x)))))
    feasible = [(value, p) for value, p in candidates if value <= -1e6*(1-1e-6)]
    nearest = min(feasible, key=lambda pair: norm_e(pair[1])) if feasible else None
    return dict(model=row['model'], seed=row['seed'], slug=row['slug'],
                model_sha256=row['model_sha256'], cloud_minimum_Pa=float(v.min()),
                cloud_negative_states=int((v<0).sum()), optimization=local_reports,
                nearest_optimization=nearest_reports, total_evaluations=len(candidates)-len(v),
                most_negative=witness(model, scales, most_negative[1]),
                nearest=witness(model, scales, nearest[1]) if nearest else None)


def main():
    torch.set_num_threads(2)
    lock, scales = verify_gate()
    OUTPUT.mkdir(exist_ok=False)
    _atomic_json(OUTPUT/'specification.json', dict(rule_sha256=digest(RULE),
        source_sha256=digest(Path(__file__)), helper_sha256=digest(Path(__file__).with_name('audit_robustness.py')),
        checkpoint_manifest_sha256=digest(LOCK), started_unix=time.time()))
    rows = []
    for row in lock['entries']:
        if row['model'] != 'Free':
            continue
        result = run_search(row, scales)
        rows.append(result)
        _atomic_json(OUTPUT/(row['slug']+'.json'), result)
        print(row['slug'], 'nearest', result['nearest'], flush=True)
    comparisons = []
    for row in rows:
        w = row['nearest']
        if not w or not w['negative_verified']:
            continue
        responses = []
        for entry in lock['entries']:
            model = load_model(entry)
            e = np.array([w['strain']])
            _, s, d = predict(model, e, scales)
            f = sqrt_c(e)
            a, b = np.array(w['a']), np.array(w['b'])
            H = np.outer(a, b)
            ed = (f[0].T@H+H.T@f[0])/2
            ev = np.array([ed[0, 0], ed[1, 1], 2*ed[0, 1]])
            S = np.array([[s[0, 0], s[0, 2]], [s[0, 2], s[0, 1]]])
            val = float(ev@d[0]@ev+np.sum(S*(H.T@H)))
            responses.append(dict(slug=entry['slug'], curvature_Pa=val))
        comparisons.append(dict(witness_slug=row['slug'], models=responses))
    verify_gate()
    _atomic_json(OUTPUT/'summary.json', dict(status='neural_search_complete',
                                           runs=rows, matched_comparisons=comparisons))


if __name__ == '__main__':
    main()
