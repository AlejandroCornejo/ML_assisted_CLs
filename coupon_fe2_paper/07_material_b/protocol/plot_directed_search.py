"""Reader-facing plots of the selected, independently verified Free witnesses."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from protocol.directed_search import OUTPUT
from protocol.audit_robustness import sqrt_c
from protocol.evaluate_final_models import verify_gate, load_model, predict


def main():
    torch.set_num_threads(2)
    lock, scales = verify_gate()
    summary = json.loads((OUTPUT/'summary.json').read_text())
    target = next(r['nearest'] for r in summary['runs'] if r['seed'] == 29)
    s = np.linspace(0, 1, 101)
    e = s[:, None]*np.array(target['strain'])
    F = sqrt_c(e)
    H = np.outer(target['a'], target['b'])
    Edot = (F.transpose(0, 2, 1)@H+H.T@F)/2
    ev = np.column_stack((Edot[:, 0, 0], Edot[:, 1, 1], 2*Edot[:, 0, 1]))
    hth = H.T@H
    curves = {}
    for row in lock['entries']:
        model = load_model(row)
        _, stress, tangent = predict(model, e, scales)
        value = np.einsum('ni,nij,nj->n', ev, tangent, ev)
        value += stress[:, 0]*hth[0, 0]+stress[:, 1]*hth[1, 1]+2*stress[:, 2]*hth[0, 1]
        curves[row['slug']] = value
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = ['#333333', '#1878b4', '#e18016', '#278a56', '#9948a6']
    names = list(dict.fromkeys(r['model'] for r in lock['entries']))
    for color, name in zip(colors, names):
        rows = [r for r in lock['entries'] if r['model'] == name]
        values = np.array([curves[r['slug']]/1e6 for r in rows])
        median_seed = 47 if name == 'ICKAN-learned' else 29
        median_row = next(r for r in rows if r['seed'] == median_seed)
        axs[0].plot(s, curves[median_row['slug']]/1e6, color=color, label=name)
        axs[0].fill_between(s, values.min(axis=0), values.max(axis=0), color=color, alpha=.12)
    eend = np.array(target['strain'])
    lo, hi = np.array([-.04, -.04, -.08]), np.array([.2, .2, .08])
    edge = min((hi[i] if v > 0 else lo[i])/v for i, v in enumerate(eend) if v != 0)
    axs[0].axvspan(0, edge, color='#d4e8cd', alpha=.4)
    axs[0].axvline(edge, color='gray', linestyle=':', linewidth=1)
    axs[0].axhline(0, color='black', linewidth=.8)
    axs[0].set(xlabel=r'Loading parameter $s$: $e(s)=s\,e_*$',
               ylabel='Curvature in the selected rank-one direction [MPa]',
               title='Same state and direction for every model')
    axs[0].legend(loc='lower left', framealpha=.85, fontsize=8)
    axs[0].text(.03, .97, 'Shaded strip: approved strain box',
                transform=axs[0].transAxes, va='top', fontsize=8)
    ts = np.linspace(-1e-3, 1e-3, 101)
    remainders = {}
    for run, color in zip(summary['runs'], ['#1878b4', '#e18016', '#278a56']):
        w = run['nearest']
        row = next(r for r in lock['entries'] if r['slug'] == run['slug'])
        model = load_model(row)
        f = torch.tensor(w['F'], dtype=torch.float64)
        h = torch.tensor(np.outer(w['a'], w['b']), dtype=torch.float64)
        def energy(t):
            ft = f[None]+t[:, None, None]*h[None]
            c = ft.transpose(1, 2)@ft
            x = torch.stack(((c[:, 0, 0]-1)/2, (c[:, 1, 1]-1)/2, c[:, 0, 1]), dim=1)/scales['strain_scale']
            return model.energy(x).reshape(-1)*scales['energy_scale']
        zero = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
        w0 = energy(zero)
        slope = torch.autograd.grad(w0.sum(), zero)[0]
        t = torch.tensor(ts, dtype=torch.float64)
        residual = (energy(t)-w0.detach()-slope.detach()*t).detach().numpy()
        remainders[run['slug']] = residual
        axs[1].plot(ts*1e3, residual, color=color, label='Free seed '+str(run['seed']))
    axs[1].axhline(0, color='black', linewidth=.8)
    axs[1].set(xlabel=r'Rank-one perturbation $t$ [$10^{-3}$]',
               ylabel=r'$W(F_*+tH)-W(F_*)-t\,W\prime(0)$ [Pa]',
               title='Each Free seed: energy curves downward')
    axs[1].legend(loc='lower center', framealpha=.85, fontsize=8)
    for ax in axs:
        ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'witnesses.png', dpi=180)
    fig.savefig(OUTPUT/'witnesses.pdf')
    plt.close(fig)
    np.savez_compressed(OUTPUT/'plot_data.npz', load_parameter=s, perturbation=ts,
        **{'curvature_'+k:v for k,v in curves.items()},
        **{'energy_remainder_'+k:v for k,v in remainders.items()})


if __name__ == '__main__':
    main()
