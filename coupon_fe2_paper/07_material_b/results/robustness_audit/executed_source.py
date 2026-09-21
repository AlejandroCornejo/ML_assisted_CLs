"""Bounded, post-test audit; see ROBUSTNESS_AUDIT_INTERNAL.md."""
from itertools import product
from pathlib import Path
import json
import time

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import qmc

from protocol.evaluate_final_models import verify_gate, load_model, predict, BASE, LOCK
from protocol.prepare_design import digest
from protocol.train_material_b import _atomic_json

OUTPUT = BASE / 'results/robustness_audit'
RULE = Path(__file__).with_name('ROBUSTNESS_AUDIT_INTERNAL.md')


def sqrt_c(e):
    e = np.asarray(e).reshape(-1, 3)
    c = np.zeros((len(e), 2, 2))
    c[:, 0, 0], c[:, 1, 1] = 1+2*e[:, 0], 1+2*e[:, 1]
    c[:, 0, 1] = c[:, 1, 0] = e[:, 2]
    vals, vecs = np.linalg.eigh(c)
    if np.min(vals) <= 0:
        raise ValueError('Nonpositive C')
    return np.einsum('nik,nk,njk->nij', vecs, np.sqrt(vals), vecs)


def acoustic(f, stress, tangent, theta):
    """Q such that a.T Q a = d2W(F)[a tensor b,a tensor b]."""
    b = np.array([np.cos(theta), np.sin(theta)])
    B = np.stack((b[0]*f[:, :, 0], b[1]*f[:, :, 1],
                  b[1]*f[:, :, 0]+b[0]*f[:, :, 1]), axis=1)
    q = np.einsum('nki,nkl,nlj->nij', B, tangent, B)
    geom = b[0]**2*stress[:, 0]+b[1]**2*stress[:, 1]+2*b[0]*b[1]*stress[:, 2]
    q += geom[:, None, None]*np.eye(2)
    vals, vecs = np.linalg.eigh((q+q.transpose(0, 2, 1))/2)
    return vals[:, 0], vecs[:, :, 0], b


def verification(model, e, a, b, scales):
    f = torch.tensor(sqrt_c(e), dtype=torch.float64)
    h = torch.tensor(np.outer(a, b)[None], dtype=torch.float64)
    ss, es = scales['strain_scale'], scales['energy_scale']
    def energy(t):
        ft = f+t*h
        c = ft.transpose(1, 2)@ft
        x = torch.stack(((c[:, 0, 0]-1)/2, (c[:, 1, 1]-1)/2, c[:, 0, 1]), dim=1)/ss
        return model.energy(x).sum()*es
    t = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    w = energy(t)
    first = torch.autograd.grad(w, t, create_graph=True)[0]
    second = float(torch.autograd.grad(first, t)[0])
    fd = {str(step): float((energy(step)-2*w+energy(-step)).detach()/step**2)
          for step in (1e-3, 3e-4, 1e-4)}
    return dict(direct_autograd_Pa=second, centered_energy_differences_Pa=fd)


def search(model, scales, factor):
    lo = factor*np.array([-.04, -.04, -.08])
    hi = factor*np.array([.20, .20, .08])
    u = np.vstack((qmc.Sobol(3, scramble=True, seed=20260918).random_base2(9),
                   np.array(list(product([0., 1.], repeat=3)))))
    e = lo+u*(hi-lo)
    _, s, d = predict(model, e, scales)
    f = sqrt_c(e)
    best_values = np.full(len(e), np.inf)
    angles = np.zeros(len(e))
    for theta in np.arange(32)*np.pi/32:
        vals, _, _ = acoustic(f, s, d, theta)
        update = vals < best_values
        best_values[update], angles[update] = vals[update], theta
    candidates, optimizers = [], []
    for idx in np.argsort(best_values)[:2]:
        x0 = np.r_[u[idx], angles[idx]]
        candidates.append((float(best_values[idx]), x0))
        def objective(x):
            strain = (lo+x[:3]*(hi-lo))[None]
            _, st, dt = predict(model, strain, scales)
            value = float(acoustic(sqrt_c(strain), st, dt, x[3])[0][0])
            candidates.append((value, x.copy()))
            return value/1e9
        result = minimize(objective, x0, method='Powell',
                          bounds=[(0., 1.)]*3+[(0., np.pi)],
                          options=dict(maxfev=120, xtol=1e-5, ftol=1e-6))
        optimizers.append(dict(success=bool(result.success), message=str(result.message),
                               nfev=int(result.nfev)))
    value, x = min(candidates, key=lambda pair: pair[0])
    strain = (lo+x[:3]*(hi-lo))[None]
    _, st, dt = predict(model, strain, scales)
    fv = sqrt_c(strain)
    _, av, bv = acoustic(fv, st, dt, x[3])
    return dict(factor=factor, lower=lo.tolist(), upper=hi.tolist(), cloud_states=len(e),
                b_directions=32, cloud_minimum_Pa=float(best_values.min()),
                cloud_negative_states=int(np.count_nonzero(best_values < 0)),
                optimizers=optimizers, minimum_Pa=value, strain=strain[0].tolist(),
                F=fv[0].tolist(), a=av[0].tolist(), b=bv.tolist(),
                verification=verification(model, strain, av[0], bv, scales))


def collapse(model, scales, free):
    j = np.logspace(0, -8, 81)
    e = np.column_stack(((j-1)/2, (j-1)/2, np.zeros_like(j)))
    x = torch.tensor(e/scales['strain_scale'], dtype=torch.float64)
    w = (model.energy(x).reshape(-1)*scales['energy_scale']).detach().numpy()
    if not np.isfinite(w).all():
        raise FloatingPointError('Nonfinite collapse energy')
    result = dict(J=j.tolist(), energy_Pa=w.tolist())
    if free:
        features = torch.tensor([[-1., -1., 0., -1.]], dtype=torch.float64)
        raw0, s0 = model.reference_terms(create_graph=False)
        xlimit = torch.tensor([[-.5, -.5, 0.]], dtype=torch.float64)/model.strain_scale
        limit = model.base_energy(features/model.feature_scale)-raw0-(s0*xlimit).sum()
        result['analytic_finite_limit_Pa'] = float(limit.detach()*scales['energy_scale'])
    return result


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = ['#333333', '#1878b4', '#e18016', '#278a56', '#9948a6']
    names = list(dict.fromkeys(r['model'] for r in rows))
    for color, name in zip(colors, names):
        group = [r for r in rows if r['model'] == name]
        for i, r in enumerate(group):
            c = r['collapse']
            axs[0].plot(c['J'], np.array(c['energy_Pa'])/1e6, color=color,
                        alpha=.85, label=name if i == 0 else None)
        for factor, marker in [(1, 'o'), (2, '^')]:
            vals = [next(s for s in r['search'] if s['factor'] == factor)['minimum_Pa']/1e6 for r in group]
            xpos = names.index(name)+(factor-1.5)*.15
            axs[1].scatter(np.full(len(vals), xpos), vals, color=color, marker=marker, s=40)
    axs[0].set(xscale='log', yscale='symlog', xlabel='Area ratio J (collapse to the right)',
               ylabel='Energy density [MPa]', title=r'Volume collapse: $F=\sqrt{J}\,I$')
    axs[0].invert_xaxis()
    axs[0].legend(loc='upper left', framealpha=.8, fontsize=8)
    axs[1].axhline(0, color='black', linewidth=.8)
    axs[1].set(xticks=range(len(names)), xticklabels=names, ylabel='Lowest found rank-one curvature [MPa]',
               title='Bounded search, all three seeds')
    axs[1].tick_params(axis='x', rotation=25)
    for factor, marker in [(1, 'o'), (2, '^')]:
        axs[1].scatter([], [], marker=marker, color='gray', label='Approved box' if factor == 1 else 'Double-size box (exploratory)')
    axs[1].legend(loc='upper left', framealpha=.8, fontsize=8)
    for ax in axs:
        ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'audit.png', dpi=180)
    fig.savefig(OUTPUT/'audit.pdf')
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    lock, scales = verify_gate()
    OUTPUT.mkdir(exist_ok=False)
    _atomic_json(OUTPUT/'specification.json', dict(rule_sha256=digest(RULE),
                 executable_sha256=digest(Path(__file__)), lock_sha256=digest(LOCK),
                 status='specified_before_execution', unix_time=time.time()))
    rows = []
    for row in lock['entries']:
        start = time.monotonic()
        model = load_model(row)
        result = {k: row[k] for k in ('model', 'seed', 'slug', 'model_sha256')}
        result['collapse'] = collapse(model, scales, row['model'] == 'Free')
        result['search'] = [search(model, scales, factor) for factor in (1, 2)]
        result['seconds'] = time.monotonic()-start
        rows.append(result)
        _atomic_json(OUTPUT/(row['slug']+'.json'), result)
        print(row['slug'], [round(s['minimum_Pa']/1e6, 3) for s in result['search']],
              f"MPa, {result['seconds']:.1f}s", flush=True)
    verify_gate()
    _atomic_json(OUTPUT/'summary.json', dict(status='neural_audit_complete', models=rows))
    plot(rows)


if __name__ == '__main__':
    main()
