#!/usr/bin/env python3
"""Regenerate manuscript tables and vector figures from frozen result artifacts.

No training, constitutive solve, or timing run is performed. Archived data are
read-only. Tables use unweighted array norms, not continuum L2 norms.
"""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / '.pydeps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# LaTeX typography shared with the other figure builders (e.g. build_common_path_evidence.py).
STYLE = {'text.usetex': True, 'text.latex.preamble': r'\usepackage{lmodern}\usepackage{amsmath}\usepackage{bm}',
         'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.65}
FIG = HERE / 'figures'
TAB = HERE / 'tables'
SOURCES = {}
SC = ROOT / '06_pann/results'
MC = ROOT / '07_material_b/results'
FE2 = ROOT / '06_fe2'
UNCONSTRAINED = 'Unconstrained energy'


def source(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{stem}.{ext}', bbox_inches='tight', dpi=220)
    plt.close(fig)


def table(name, headings, rows, columns, note=''):
    text = '\\begin{tabular}{' + columns + '}\n\\toprule\n'
    text += ' & '.join(headings) + ' \\\\\n\\midrule\n'
    text += ''.join(' & '.join(row) + ' \\\\\n' for row in rows)
    text += '\\bottomrule\n\\end{tabular}\n'
    if note:
        text += '\n\\par\\smallskip{\\footnotesize ' + note + '}\n'
    (TAB / name).write_text(text)


def selected_checkpoints(path, key):
    """Validation selections frozen before the test and probe gates, with verified hashes."""
    selection = read_json(path)
    assert selection['status'] == 'frozen_before_test_probe'
    assert selection['test_probe_accessed'] is False
    rows = {}
    for row in selection['selected']:
        checkpoint = source(Path(row['checkpoint']))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_sha256']
        rows[key(row)] = dict(row, path=checkpoint)
    return rows


def audit_entry(audit, checkpoint):
    for key, value in audit.items():
        path = Path(key) if Path(key).is_absolute() else ROOT / key
        if key != 'gate' and path.resolve() == Path(checkpoint).resolve():
            return value
    raise KeyError(checkpoint)


def coupon_run(stem, checkpoint_sha256):
    """Final nodal displacement error of one archived coupon run against the nested FOM."""
    meta = read_json(FE2 / (stem + '.json'))
    reference_meta = read_json(FE2 / 'fom_fe2_clean_timing_fomfe2_w4_f100kn.json')
    assert meta['status'] == 'converged' and meta['n_steps_completed'] == 20
    assert meta['checkpoint']['checkpoint_sha256'] == checkpoint_sha256, stem
    assert all(step['coverage']['outside'] == 0 for step in meta['macro_newton'])
    for key in ('macro_mesh_sha256', 'rve_mesh_sha256', 'force_per_end'):
        assert meta[key] == reference_meta[key], (stem, key)
    with np.load(source(FE2 / (stem + '.npz'))) as raw, \
            np.load(source(FE2 / 'fom_fe2_clean_timing_fomfe2_w4_f100kn.npz')) as ref:
        error = np.linalg.norm(raw['u_nodal'] - ref['u_nodal']) / np.linalg.norm(ref['u_nodal'])
    return 100 * float(error), sorted({step['iterations'] for step in meta['macro_newton']})


def trainable_parameters(checkpoint):
    import torch
    sys.path.insert(0, str(ROOT / '06_pann'))
    from flexible_pann import load_flexible
    model, _ = load_flexible(checkpoint)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def finite_element_results():
    path = ROOT / '06_fe2'
    cases = [
        ('FOM', 'fom_fe2_clean_timing_fomfe2_w4_f100kn', 1),
        ('HPROM', 'hprom_fe2_clean_timing_ecm_w20_f100kn_optimized_r', 3),
        ('HPROM--ANN', 'hprom_ann_fe2_clean_timing_maw10_w20_f100kn_optimized_r', 3),
        ('D-HPROM--ANN', 'dhprom_ann_fe2_clean_timing_maw10_direct_w20_f100kn_optimized_r', 3),
        ('ICNN', 'pann_fe2_clean_timing_icnn6_t2_f100kn', 1),
        ('ICKAN', 'pann_fe2_clean_timing_ickan6_t8_f100kn', 1),
        (UNCONSTRAINED, 'pann_fe2_clean_timing_free_sp_t8_f100kn', 1),
    ]
    # The PANN rows must use the validation-selected checkpoints of the constitutive study.
    constrained = selected_checkpoints(SC / 'sc_m06_learned_v1/training/validation_selection.json',
                                       lambda row: row['core'])
    expected = {'ICNN': constrained['ICNN']['checkpoint_sha256'],
                'ICKAN': constrained['ICKAN']['checkpoint_sha256'],
                UNCONSTRAINED: hashlib.sha256(source(SC / 'sc_unconstrained_v1/fe2_free_checkpoint.pt')
                                              .read_bytes()).hexdigest()}
    results = []
    reference = None
    reference_meta = None
    for name, stem, repeats in cases:
        stems = [stem] if repeats == 1 else [stem + str(i) for i in (1, 2, 3)]
        summaries = [read_json(path / (s + '.json')) for s in stems]
        times = [s['wall_seconds'] for s in summaries]
        med_index = int(np.argsort(times)[len(times)//2])
        meta = summaries[med_index]
        with np.load(source(path / (stems[med_index] + '.npz'))) as raw:
            data = {k: raw[k].copy() for k in raw.files}
        if reference is None:
            reference, reference_meta = data, meta
        if name in expected:
            assert meta['checkpoint']['checkpoint_sha256'] == expected[name], name
        for m in summaries:
            assert m['status'] == 'converged' and m['n_steps_completed'] == 20
            for key in ('macro_mesh_sha256', 'rve_mesh_sha256', 'force_per_end'):
                assert m[key] == reference_meta[key], (name, key)
            assert [s['iterations'] for s in m['macro_newton']] == [4]*20
            assert all(s['coverage']['outside'] == 0 for s in m['macro_newton'])
        for key in ('coords', 'connectivity'):
            assert np.array_equal(data[key], reference[key]), (name, key)
        errors = {k: 100*float(np.linalg.norm(data[k]-reference[k])/np.linalg.norm(reference[k]))
                  for k in ('u_nodal', 'E_final', 'S_final')}
        results.append(dict(model=name, stems=stems, times=times,
                            seconds=float(np.median(times)),
                            speedup=reference_meta['wall_seconds']/float(np.median(times)),
                            concurrency=meta.get('workers', meta.get('torch_threads')),
                            concurrency_kind='processes' if 'workers' in meta else 'threads',
                            repeats=repeats, errors_percent=errors))
    # All tables: percentages with four decimals; times and speed-ups with two.
    rows = []
    for row in results:
        errors = row['errors_percent']
        is_reference = row['model'] == 'FOM'
        cells = ['--']*3 if is_reference else [f'{errors[k]:.4f}' for k in errors]
        rows.append([row['model'], *cells, f"{row['seconds']:.2f}",
                     '--' if is_reference else f"{row['speedup']:.2f}"])
    table('fe2_comparison.tex', ['Model', '$e_u$ [\\%]', '$e_E$ [\\%]', '$e_S$ [\\%]',
          'Time [s]', 'Speed-up'], rows, 'lrrrrr')
    field_figures(reference)
    return results


# Same field style as the RVE field figure (build_common_path_evidence.py): coolwarm for the
# displacement magnitude, jet for nodally averaged stresses, thin slate mesh lines.
COUPON_CMAPS = {'u': 'coolwarm', 'S': 'jet'}
FIELD_ALPHA = 0.82
MESH_LINES = dict(color='#334155', linewidth=0.075, alpha=0.30)


def coupon_mesh(data):
    """Deformed coupon triangulations and the element-to-node recovery used in every coupon figure."""
    t = data['connectivity'].astype(int)
    xy = 1000*(data['coords']+data['u_nodal'])
    fine = np.vstack([t[:, cols] for cols in ([0,3,5], [3,1,4], [5,4,2], [3,4,5])])
    tri = mtri.Triangulation(xy[:,0], xy[:,1], fine)
    coarse = mtri.Triangulation(xy[:,0], xy[:,1], t[:,:3])
    def recover(values):
        vals = values.reshape(-1,3).mean(axis=1)
        out, count = np.zeros(len(xy)), np.zeros(len(xy))
        np.add.at(out, t.ravel(), np.repeat(vals,6))
        np.add.at(count, t.ravel(), 1)
        return out/np.maximum(count,1)
    return tri, coarse, recover


def cauchy_equivalent(E, S):
    """In-plane von Mises equivalent of the Cauchy stress from Green strain (engineering shear) and PK2 stress."""
    C = np.empty((len(E), 2, 2))
    C[:, 0, 0], C[:, 1, 1], C[:, 0, 1], C[:, 1, 0] = 1 + 2*E[:, 0], 1 + 2*E[:, 1], E[:, 2], E[:, 2]
    w, v = np.linalg.eigh(C)
    U = np.einsum('nij,nj,nkj->nik', v, np.sqrt(w), v)     # the rotation of F = RU drops out of the invariant
    P = np.empty((len(S), 2, 2))
    P[:, 0, 0], P[:, 1, 1], P[:, 0, 1], P[:, 1, 0] = S[:, 0], S[:, 1], S[:, 2], S[:, 2]
    sigma = U @ P @ U / np.sqrt(np.linalg.det(C))[:, None, None]
    s11, s22, s12 = sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 0, 1]
    return np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)


def field_figures(data):
    tri, coarse, recover = coupon_mesh(data)
    rows = [(r'$\|u\|$ [mm]', 1000*np.linalg.norm(data['u_nodal'], axis=1), COUPON_CMAPS['u'], 0.0),
            (r'$\sigma_{\rm eq}$ [MPa]', recover(cauchy_equivalent(data['E_final'], data['S_final'])/1e6),
             COUPON_CMAPS['S'], None)]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 1.85), layout='constrained')
    fig.get_layout_engine().set(h_pad=0.03, hspace=0.0)
    for ax, (label, value, cmap, vmin) in zip(axes, rows):
        im = ax.tripcolor(tri, value, shading='gouraud', cmap=cmap, vmin=vmin, alpha=FIELD_ALPHA)
        ax.triplot(coarse, **MESH_LINES)
        ax.set_aspect('equal')
        ax.margins(x=0.012, y=0.06)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#7C838C'); spine.set_linewidth(0.6)
        ax.set_ylabel(label)
        # The colorbar follows the drawn (equal-aspect) frame, so both have the same height.
        cbar = fig.colorbar(im, cax=ax.inset_axes([1.012, 0.0, 0.013, 1.0]))
        cbar.ax.tick_params(labelsize=7)
    save(fig, 'coupon_fields')
    two_scale_figure(data, tri, coarse, recover)


def two_scale_figure(data, tri, coarse, recover):
    """Undeformed coupon with its boundary conditions, and the SC-RVE queried at every integration point."""
    from matplotlib.patches import Circle, ConnectionPatch, Ellipse, FancyArrowPatch
    from matplotlib.collections import PolyCollection
    sys.path.insert(0, str(HERE))
    from build_microstructure_figures import cavities_a, load_a_mesh, material_a
    red, zoom = '#c44939', '#c0392b'
    fill, line = '#DCE8F1', '#4F6A80'          # the cell colours of the RVE geometry figures
    fig = plt.figure(figsize=(7.1, 2.45))
    ax = fig.add_axes([0.0, 0.02, 0.63, 0.96])
    xy = 1000*data['coords']
    t = data['connectivity'].astype(int)[:, :3]
    ax.add_collection(PolyCollection(xy[t], facecolors=fill, edgecolors=line, linewidths=0.2))
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    tol = 1e-6
    left, right = np.flatnonzero(np.abs(xy[:, 0] - x0) < tol), np.flatnonzero(np.abs(xy[:, 0] - x1) < tol)
    half = xy[left, 1].max()
    for face, x, sign in ((left, x0, -1), (right, x1, 1)):
        for y in (-0.88*half, -0.52*half, 0.52*half, 0.88*half):   # mid-height is left to the constraint label
            ax.add_patch(FancyArrowPatch((x + 0.8*sign, y), (x + 9*sign, y), arrowstyle='-|>',
                                         mutation_scale=7, color=red, lw=0.9))
    ax.text(x0 - 5, half + 2.2, '100 kN per end', ha='left', va='bottom', fontsize=8, color=red)
    pin = left[np.argmin(np.abs(xy[left, 1]))]
    roller = right[np.argmin(np.abs(xy[right, 1]))]
    # Each constraint is labelled horizontally beside its node, between the load arrows.
    for node, text, sign in ((pin, r'$\bm u=\bm 0$', -1), (roller, r'$u_2=0$', 1)):
        ax.plot(*xy[node], 'o', color='black', ms=3.2, zorder=6)
        ax.text(xy[node, 0] + 2.8*sign, xy[node, 1], text, fontsize=8, va='center',
                ha='right' if sign < 0 else 'left')
    # Low enough that the lower zoom line passes above the right end of the dimension line.
    ax.annotate('', xy=(x0, -half - 16.5), xytext=(x1, -half - 16.5), arrowprops=dict(arrowstyle='<->', lw=0.7))
    ax.text(0.5*(x0 + x1), -half - 18, '165 mm', ha='center', va='top', fontsize=8.5)
    point = np.array([12.0, 0.0])
    ax.add_patch(Circle(point, 3.0, fill=False, ec=zoom, lw=1.0, zorder=5))
    ax.set_aspect('equal'); ax.set_xlim(x0 - 21, x1 + 14); ax.set_ylim(-half - 24, half + 8); ax.axis('off')
    rx = fig.add_axes([0.66, 0.03, 0.32, 0.94])
    rxy, rconn = load_a_mesh(); side = material_a.CELL_SIDE
    scale = 0.8/side                                   # the cell and its dimension fit inside the circle
    rx.add_collection(PolyCollection((rxy*scale)[rconn[:, :3]], facecolors=fill, edgecolors=line, linewidths=0.2))
    for cx, cy, a, b, angle in cavities_a():
        rx.add_patch(Ellipse((cx*scale, cy*scale), 2*a*scale, 2*b*scale, angle=angle, fc='white',
                             ec='#334155', lw=0.8))
    rx.add_patch(Circle((0, 0), 0.74, fill=False, ec=zoom, lw=1.3, clip_on=False))
    rx.annotate('', xy=(-0.4, -0.49), xytext=(0.4, -0.49), arrowprops=dict(arrowstyle='<->', lw=0.7))
    rx.text(0, -0.53, r'$\ell=1.3$ mm', ha='center', va='top', fontsize=8.5)
    rx.set_xlim(-0.78, 0.78); rx.set_ylim(-0.78, 0.78); rx.set_aspect('equal'); rx.axis('off')
    for angle in (np.radians(150), np.radians(210)):
        fig.add_artist(ConnectionPatch(xyA=(point[0], point[1] + 3.0*np.sign(np.sin(angle))), coordsA=ax.transData,
                                       xyB=(0.74*np.cos(angle), 0.74*np.sin(angle)), coordsB=rx.transData,
                                       color=zoom, lw=0.7))
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'two_scale_coupon.{ext}', dpi=220, facecolor='white')
    plt.close(fig)


def constitutive_results():
    audit = read_json(ROOT/'06_pann/results/sc_m06_learned_v1/independent_audit.json')
    selection = read_json(ROOT/'06_pann/results/sc_m06_learned_v1/training/validation_selection.json')
    assert selection['status'] == 'frozen_before_test_probe'
    assert selection['test_probe_accessed'] is False
    assert {row['core']: row['seed'] for row in selection['selected']} == {'ICNN': 16, 'ICKAN': 16}
    for row in selection['selected']:
        checkpoint = source(Path(row['checkpoint']))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_sha256']
    unconstrained = selected_checkpoints(SC / 'sc_unconstrained_v1/training/validation_selection.json',
                                         lambda row: row['core'])['Free']
    unconstrained_audit = read_json(SC / 'sc_unconstrained_v1/independent_audit.json')
    assert unconstrained_audit['gate']['status'] == 'opened_once'
    entries = [('ICNN', next(v for k, v in audit.items() if 'm06_icnn' in k), MODEL_COLORS['ICNN']),
               ('ICKAN', next(v for k, v in audit.items() if 'm06_ickan' in k), MODEL_COLORS['ICKAN']),
               (UNCONSTRAINED, audit_entry(unconstrained_audit, unconstrained['path']), MODEL_COLORS['Unconstrained'])]
    rows, fig = [], plt.figure(figsize=(6.9,3.4),layout='constrained')
    ax = fig.subplots()
    for name, entry, color in entries:
        m = entry['metrics']
        rows.append([name, f"{100*m['test']['stress']:.4f}", f"{100*m['test']['energy']:.4f}",
                     f"{100*m['probe']['stress']:.4f}",
                     f"{100*m['test']['stress_sample_relative_percentiles'][-1]:.4f}",
                     f"{100*m['probe']['stress_sample_relative_percentiles'][-1]:.4f}"])
        rings = entry['probe_by_ring']
        ax.plot([float(r) for r in rings],[100*rings[r]['stress'] for r in rings], 'o-', label=name,color=color)
    # Probe and worst-state evidence stays in the internal supplement, which the
    # paper does not cite; do not restore those columns in the main-paper table.
    test_rows = [[row[0], row[1], row[2]] for row in rows]
    table('constitutive_errors.tex',
          ['Model', '$e_S^{\\rm test}$ [\\%]', '$e_W^{\\rm test}$ [\\%]'],
          test_rows, 'lrr',
          'All values are percentages. Stress and energy errors are aggregate array norms. '
          'SC-RVE: 400 independent test states.')
    table('constitutive_probe_errors.tex',['Model', '$e_S^{\\rm test}$ [\\%]', '$e_W^{\\rm test}$ [\\%]', '$e_S^{\\rm probe}$ [\\%]',
          'Worst test [\\%]', 'Worst probe [\\%]'],rows,'lrrrrr',
          'All values are percentages. The last two columns are maximum per-state stress errors; '
          'the first three are aggregate array norms. Test: 400 states; probe: 345 finite reference states out of 350.')
    ax.set(xlabel='Sampling-box overshoot factor (not a load factor)',ylabel='Probe stress error [\\%]',
           yscale='log')
    ax.legend(frameon=False); ax.grid(alpha=.2)
    save(fig,'probe_errors')


def mechanics():
    historical_cycle=read_json(ROOT/'06_pann/mechanics_witness_results/current_rve_cycle_audit.json')['cycle']
    # Unconstrained entries recomputed on the states, directions and cycle of the m=6 audit;
    # the ICNN and ICKAN entries are copied from it and reproduced by the audit itself.
    current_audit=read_json(ROOT/'06_pann/results/sc_unconstrained_v1/mechanics_audit.json')
    for name, record in current_audit['control_constrained_reproduction'].items():
        assert record['recomputed_Pa'] == record['stored_Pa'], name
    cycle=current_audit['cycle']
    cycle['FOM']=historical_cycle['FOM']
    conv=cycle['quadrature_work_convergence_J_per_m3']
    maw=read_json(ROOT/'06_fe2/maw_closed_cycle_audit.json')
    assert maw['status'] == 'completed'
    assert maw['centre_E'] == cycle['centre_E']
    assert maw['half_width_in_E11_and_E22'] == cycle['half_width_in_E11_and_E22']
    for relative, expected in maw['source_sha256'].items():
        assert hashlib.sha256(source(ROOT/relative).read_bytes()).hexdigest() == expected
    assert set(maw['convergence']) == set(conv)
    for order in conv:
        conv[order]['HPROM--ANN'] = maw['convergence'][order]['signed_work_J_per_m3']
    other=read_json(ROOT/'06_fe2/other_hprom_closed_cycle_audit.json')
    assert other['status'] == 'completed'
    assert other['centre_E'] == cycle['centre_E']
    assert other['half_width_in_E11_and_E22'] == cycle['half_width_in_E11_and_E22']
    for relative, expected in other['source_sha256'].items():
        assert hashlib.sha256(source(ROOT/relative).read_bytes()).hexdigest() == expected
    for name, record in other['models'].items():
        assert record['status'] == 'completed'
        assert set(record['convergence']) == set(conv)
        for order in conv:
            conv[order][name] = record['convergence'][order]['signed_work_J_per_m3']
    def sci(value):
        mantissa, exponent = f'{value:.2e}'.split('e')
        return f'${mantissa}\\times10^{{{int(exponent)}}}$'
    order = '128'
    cycle_rows = [['FOM (8 per edge)', sci(cycle['FOM']['stress_work_J_per_m3'])]]
    cycle_rows += [[label, sci(conv[order][name])] for name, label in
                   [('Free', UNCONSTRAINED), ('ICNN', 'ICNN'), ('ICKAN', 'ICKAN'), ('HPROM', 'HPROM'),
                    ('HPROM--ANN', 'HPROM--ANN'), ('D-HPROM--ANN', 'D-HPROM--ANN')]]
    table('cycle_work.tex', ['Model', '$\\mathcal W_{\\rm cyc}$'], cycle_rows, 'lr',
          f'{order} Gauss points per edge, except for the FOM. '
          'The ICKAN residual decreases with quadrature refinement through its spline knots.')
    audit=current_audit
    rows=[]
    for key,label in [('held_out_test','Held-out test'),('uniform_training_box','In-box states'),
                      ('converged_probe','Finite-label probe'),
                      ('radial_box_expansion','Radial expansion, up to $4\\times$ the box'),
                      ('broad_unvalidated_principal_stretch_scan','Principal stretches $0.25$--$3$')]:
        cloud=audit['rank_one']['clouds'][key]
        rows.append([label,f"{cloud['models']['Free']['n_states']:,}".replace(',', '\\,'),
                     *(f"${cloud['models'][n]['curvature_Pa']/1e6:.1f}$" for n in ('Free','ICNN','ICKAN'))])
        assert all(v['n_b_directions']==180 for v in cloud['models'].values())
        assert all(cloud['models'][n]['curvature_Pa'] > 0 for n in ('ICNN','ICKAN'))
    table('rank_one.tex',['State set','$N$','Unconstrained','ICNN','ICKAN'],rows,'lrrrr',
          'Minimum sampled rank-one curvature in MPa, over 180 unit directions $\\bm b$ with exact '
          'minimization over $\\bm a$. Minima need not occur at the same state or direction. '
          'The last two sets lie beyond the data.')


def thin(value):
    return f'{value:,}'.replace(',', '\\,')


def capacity_controls():
    # MC-RVE: medians over three seeds, as in Table 3.
    m06 = read_json(MC / 'feature_count_analysis_v1/m06_independent_evaluation_v1/summary.json')
    per_run = read_json(MC / 'feature_count_analysis_v1/m06_independent_evaluation_v1/per_run.json')
    gate_a = read_json(MC / 'capacity_2x2_v1/evaluation_gateA.json')
    gate_b = read_json(MC / 'capacity_2x2_v1/evaluation_gateB.json')
    rule = read_json(MC / 'capacity_2x2_v1/training_rule.json')
    assert gate_a['gate']['status'] == gate_b['gate']['status'] == 'opened_once'
    small = {core: trainable_parameters(source(MC / 'feature_count_amendment_v2/training'
                                               / f'm06_{core.lower()}_learned_seed16/model.pt'))
             for core in ('ICNN', 'ICKAN')}
    assert {row['slug'] for row in per_run['rows']} >= {'m06_icnn_learned_seed16', 'm06_ickan_learned_seed16'}
    cells = [('ICNN, $m=6$', small['ICNN'], m06['by_model']['ICNN-learned']['test_aggregate_percent']),
             ('ICNN, wide', rule['new_cells']['icnn_large']['trainable_parameters'],
              {k: v for k, v in gate_a['summary']['icnn_large']['test'].items()}),
             ('ICKAN, $m=6$', small['ICKAN'], m06['by_model']['ICKAN-learned']['test_aggregate_percent']),
             ('ICKAN, wide$^\\ast$', rule['new_cells']['ickan_large']['trainable_parameters'],
              gate_b['summary']['ickan_large']['test']),
             ('Unconstrained, small', rule['new_cells']['free_small']['trainable_parameters'],
              gate_a['summary']['free_small']['test']),
             ('Unconstrained', rule['new_cells']['free_large']['trainable_parameters'],
              gate_a['summary']['free_large']['test'])]
    stops = [read_json(MC / f'capacity_2x2_v1/training/ickan_large_seed{seed}/run_report.json')
             for seed in (16, 29, 47)]
    early = sorted(r['adam_steps'] for r in stops if r['adam_stop_reason'] == 'validation_plateau')
    # The note stays true only if two seeds met the plateau far earlier than the third.
    assert len(early) == 3 and early[1] < early[2] / 3
    table('capacity_mc.tex', ['Model', 'Parameters', 'Stress [\\%]', 'Energy [\\%]', 'Tangent [\\%]'],
          [[name, thin(count),
            *(f"{metrics[k]['median']:.4f}" for k in ('stress', 'energy', 'tangent'))]
           for name, count, metrics in cells], 'lrrrr',
          'Relative error norm over the 512 test states, median over three seeds. '
          '$^\\ast$Two of the three seeds stopped early at the validation plateau.')
    # SC-RVE: the validation-selected seed of every cell, as in Table 4.
    m06_sc = selected_checkpoints(SC / 'sc_m06_learned_v1/training/validation_selection.json',
                                  lambda row: row['core'])
    wide = {**selected_checkpoints(SC / 'sc_capacity_2x2_v1/training/validation_selection_gateA.json',
                                   lambda row: row['cell']),
            **selected_checkpoints(SC / 'sc_capacity_2x2_v1/training/validation_selection_gateB.json',
                                   lambda row: row['cell'])}
    m32 = selected_checkpoints(SC / 'sc_m32_learned_v1/training/validation_selection.json',
                               lambda row: 'ICNN' if 'icnn' in row['slug'] else 'ICKAN')
    free = selected_checkpoints(SC / 'sc_unconstrained_v1/training/validation_selection.json',
                                lambda row: row['core'])['Free']
    audits = {name: read_json(SC / name) for name in (
        'sc_m06_learned_v1/independent_audit.json', 'sc_capacity_2x2_v1/independent_audit_gateA.json',
        'sc_capacity_2x2_v1/independent_audit_gateB.json', 'sc_m32_learned_v1/independent_audit.json',
        'sc_unconstrained_v1/independent_audit.json')}
    def sha(path):
        return hashlib.sha256(source(path).read_bytes()).hexdigest()
    sc_cells = [
        ('ICNN, $m=6$', m06_sc['ICNN'], 'sc_m06_learned_v1/independent_audit.json',
         'pann_fe2_clean_timing_icnn6_t2_f100kn', None),
        ('ICNN, wide', wide['icnn_large'], 'sc_capacity_2x2_v1/independent_audit_gateA.json',
         'pann_fe2_capacity2x2_icnn_large_f100kn', None),
        ('ICNN, $m=32$', m32['ICNN'], 'sc_m32_learned_v1/independent_audit.json',
         'pann_fe2_capacity_m32_icnn_f100kn', None),
        ('ICKAN, $m=6$', m06_sc['ICKAN'], 'sc_m06_learned_v1/independent_audit.json',
         'pann_fe2_clean_timing_ickan6_t8_f100kn', None),
        ('ICKAN, wide', wide['ickan_large'], 'sc_capacity_2x2_v1/independent_audit_gateB.json',
         'pann_fe2_capacity2x2_ickan_large_f100kn', None),
        ('ICKAN, $m=32$', m32['ICKAN'], 'sc_m32_learned_v1/independent_audit.json',
         'pann_fe2_capacity_m32_ickan_f100kn', None),
        ('Unconstrained, small', wide['free_small'], 'sc_capacity_2x2_v1/independent_audit_gateA.json',
         'pann_fe2_capacity2x2_free_small_f100kn', SC / 'sc_capacity_2x2_v1/fe2_free_small_checkpoint.pt'),
        ('Unconstrained', free, 'sc_unconstrained_v1/independent_audit.json',
         'pann_fe2_clean_timing_free_sp_t8_f100kn', SC / 'sc_unconstrained_v1/fe2_free_checkpoint.pt')]
    rows = []
    for name, row, audit_name, stem, repackaged in sc_cells:
        entry = audit_entry(audits[audit_name], row['path'])
        test, probe = entry['metrics']['test'], entry['metrics']['probe']
        assert test['count'] == 400 and probe['count'] == 345
        if repackaged is None:
            count = trainable_parameters(row['path'])
            certificate = entry['nonnegative_energy_certificate']
            margin = certificate.get('interval_min_margin',
                                     min(certificate['small_J_margin'], certificate['large_J_slope_margin']))
            # Appendix F states that the sufficient bound fails only for the 32-feature models.
            assert (margin < 0) == ('m=32' in name), (name, margin)
            coupon, _ = coupon_run(stem, row['checkpoint_sha256'])
        else:
            count = read_json(row['path'].parent / 'run_report.json')['trainable_parameters']
            coupon, _ = coupon_run(stem, sha(repackaged))
        rows.append([name, thin(count), f"{100*test['stress']:.4f}", f'{coupon:.4f}'])
    table('capacity_sc.tex', ['Model', 'Parameters', '$e_S^{\\rm test}$ [\\%]', '$e_u$ [\\%]'], rows, 'lrrr',
          'Validation-selected seed of each cell. Test: 400 states; $e_u$: final nodal displacement error of '
          'the coupon of Section~\\ref{sec:coupon}.')


def confined_compression():
    """Oedometer test beyond the data (06_fe2/results/confined_compression_v1), 20-increment protocol."""
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
    folder = FE2 / 'results/confined_compression_v1'
    rule = read_json(folder / 'rule.json')
    executed = source(folder / 'executed_run_pann_confined_block.py')
    assert hashlib.sha256(executed.read_bytes()).hexdigest() == rule['runner_sha256']
    # Declared post-hoc diagnostic: repeated runs with identical outcomes, rank-one curvature at the Gauss points.
    audit = read_json(folder / 'rank_one/rank_one_audit.json')
    assert audit['status'] == 'complete' and all(audit['identical_records'].values())
    assert audit['script_sha256'] == hashlib.sha256(source(FE2 / 'audit_confined_rank_one.py').read_bytes()).hexdigest()
    colors = MODEL_COLORS
    runs = {(name, d): read_json(folder / f'{name}_{d}_incremental_500.json') for name in colors for d in ('x', 'y')}
    fig = plt.figure(figsize=(7.2, 4.9), layout='constrained')
    grid = fig.add_gridspec(2, 3, width_ratios=[0.72, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    # The block is a homogenized continuum: draw its actual mesh; the cavity lives in the RVE below it.
    mesh = np.load(source(folder / 'rank_one' / 'ICKAN_y_incremental_500.npz'))
    corners = 1e2*mesh['coords'][mesh['triangles'][:, :3]] + np.array([0.0, 0.5])
    fill, line, zoom = '#DCE8F1', '#4F6A80', '#c0392b'      # the colours of the coupon two-scale figure
    ax.add_collection(PolyCollection(corners, facecolors=fill, edgecolors=line, linewidths=0.35))
    for patch in (Rectangle((-0.1, -0.1), 0.1, 1.15, fc='#8a8a8a'), Rectangle((1.0, -0.1), 0.1, 1.15, fc='#8a8a8a'),
                  Rectangle((-0.1, -0.1), 1.2, 0.1, fc='#8a8a8a')):
        ax.add_patch(patch)
    for xx in np.linspace(0.15, 0.85, 4):
        ax.add_patch(FancyArrowPatch((xx, 1.32), (xx, 1.03), arrowstyle='-|>', mutation_scale=10,
                                     color='#c44939', lw=1.2))
    ax.text(0.5, 1.38, 'uniform traction $p$', ha='center', color='#c44939', fontsize=8)
    ax.text(1.15, 0.475, 'rigid, frictionless walls', rotation=90, ha='left', va='center', fontsize=7.5)
    # Zoom from an integration point to the SC-RVE, drawn as in the coupon two-scale figure.
    sys.path.insert(0, str(HERE))
    from build_microstructure_figures import cavities_a, load_a_mesh, material_a
    from matplotlib.patches import Circle
    point, radius, centre, big = np.array([0.5, 0.5]), 0.07, np.array([0.5, -0.87]), 0.55
    rxy, rconn = load_a_mesh()
    scale = 1.08*big/material_a.CELL_SIDE            # cell side over circle radius as in the coupon figure
    ax.add_collection(PolyCollection((rxy*scale + centre)[rconn[:, :3]], facecolors=fill, edgecolors=line,
                                     linewidths=0.2, zorder=3))
    for cx, cy, a, b, angle in cavities_a():
        ax.add_patch(Ellipse((cx*scale + centre[0], cy*scale + centre[1]), 2*a*scale, 2*b*scale, angle=angle,
                             fc='white', ec='#334155', lw=0.8, zorder=4))
    ax.add_patch(Circle(point, radius, fill=False, ec=zoom, lw=1.0, zorder=5))
    ax.add_patch(Circle(centre, big, fill=False, ec=zoom, lw=1.3, zorder=5))
    for sign in (-1, 1):
        end = centre + big*np.array([sign*np.cos(np.radians(45)), np.sin(np.radians(45))])
        ax.plot([point[0] + sign*radius, end[0]], [point[1], end[1]], color=zoom, lw=0.7, zorder=5)
    ax.set_xlim(-0.2, 1.3); ax.set_ylim(-1.47, 1.55); ax.set_aspect('equal'); ax.axis('off')
    ax.set_anchor('N')                                  # title level with those of (b) and (c)
    ax.set_title('(a) Setup, compression in $y$', fontsize=9)
    for column, d in enumerate(('x', 'y'), start=1):
        top = fig.add_subplot(grid[0, column])
        bottom = fig.add_subplot(grid[1, column], sharex=top)
        for name, color in colors.items():
            record = runs[(name, d)]
            shortening = [0.0] + [100*(1 - step['mean_stretch']) for step in record['steps']]
            pressure = [0.0] + [step['pressure_Pa']/1e6 for step in record['steps']]
            top.plot(shortening, pressure, '-o', color=color, ms=2.6, lw=1.2, label=name)
            if record['status'] != 'converged':
                top.plot(shortening[-1] + 1.2, pressure[-1] + 25, 'x', color=color, ms=8, mew=2.0)
            steps = audit['cases'][f'{name}_{d}_incremental_500']['steps']
            assert [s['step'] for s in steps] == [s['step'] for s in record['steps']]
            bottom.plot([100*s['shortening'] for s in steps], [s['min_curvature_MPa'] for s in steps], '-o',
                        color=color, ms=2.6, lw=1.2)
        bottom.axhline(0.0, color='black', lw=0.6)
        top.set(xlim=(0, 33), ylim=(0, 530))
        top.tick_params(labelbottom=False)
        bottom.set(xlabel='Shortening [\\%]', ylim=(-130, 430))
        for axis in (top, bottom):
            axis.grid(alpha=0.2)
        top.set_title(f'({"bc"[column - 1]}) Compression in ${d}$', fontsize=9)
        bottom.set_title(f'({"de"[column - 1]}) Compression in ${d}$', fontsize=9)
        if column == 1:
            top.set_ylabel('Applied pressure [MPa]')
            bottom.set_ylabel('Min. rank-one curvature [MPa]')
            top.legend(frameon=False, fontsize=7.5, loc='upper left')
    save(fig, 'confined_compression')
    return runs


def geometry():
    # The paired RVE panels are built together by build_microstructure_figures.py.
    # This mesh is still needed for the reduced-support visualization below.
    path=source(ROOT/'03_data/rve_mesh.mdpa')
    nodes, elements={},[]; section=None
    for line in path.read_text().splitlines():
        if line.startswith('Begin Nodes'): section='nodes'; continue
        if line.startswith('Begin Elements'): section='elements'; continue
        if line.startswith('Begin Geometries Triangle2D6'): section='triangles'; continue
        if line.startswith('End '): section=None
        vals=line.split()
        if section=='nodes' and len(vals)==4: nodes[int(vals[0])]=[float(v) for v in vals[1:3]]
        elif section=='elements' and len(vals)>=8: elements.append([int(v) for v in vals[2:8]])
        elif section=='triangles' and len(vals)==7: elements.append([int(v) for v in vals[1:7]])
    ids=list(nodes); index={v:i for i,v in enumerate(ids)}
    xy=np.asarray([nodes[i] for i in ids]); t=np.asarray([[index[i] for i in row] for row in elements])
    assert len(t)==1546
    supports_figure(xy,t)


def supports_figure(xy,t):
    folder=ROOT/'05_validation'
    fixed=np.load(source(folder/'ecm_supports.npz'))
    adaptive_r=np.load(source(folder/'maw_res_long10.npz'))['res_10_z'].astype(int)
    adaptive_s=np.load(source(folder/'maw_phase2_sig.npz'))['sig_10_z'].astype(int)
    cases=[('HPROM',fixed['z_res'].astype(int),fixed['z_sig'].astype(int),183),
           ('HPROM--ANN',adaptive_r,adaptive_s,19),
           ('D-HPROM--ANN',np.array([],dtype=int),adaptive_s,10)]
    fig,axes=plt.subplots(1,3,figsize=(7.1,3),layout='constrained')
    palette=['#f3f5f7','#315c80','#d58b28','#7754a0']
    triang=mtri.Triangulation(xy[:,0],xy[:,1],t[:,:3])
    for ax,(label,res,sig,count) in zip(axes,cases):
        tags=np.zeros(len(t)); tags[res]+=1; tags[sig]+=2
        assert np.count_nonzero(tags)==count
        ax.tripcolor(triang,facecolors=tags,cmap=ListedColormap(palette),vmin=0,vmax=3,edgecolors='#c7ced5',lw=.13)
        ax.set_aspect('equal')
        ax.set_title(f'{label}\n{len(res)} residual / {len(sig)} stress\n{count} distinct elements',fontsize=8.5)
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.legend(handles=[Patch(facecolor=palette[i],label=label) for i,label in
                        [(1,'Residual only'),(2,'Stress only'),(3,'Shared')]],
               loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    save(fig,'reduced_supports')


# Model colors shared by every figure (also in build_mc_rve_m06_evidence.py): ICNN green, ICKAN blue,
# Unconstrained red; the triple passes the dataviz palette checks, color-vision deficiency included.
MODEL_COLORS = {'ICNN': '#2AA780', 'ICKAN': '#2B5DAA', 'Unconstrained': '#C44939'}


def main():
    FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
    source(HERE/'build_evidence.py')
    plt.rcParams.update(STYLE)
    results=finite_element_results()
    constitutive_results(); mechanics(); capacity_controls(); confined_compression(); geometry()
    (HERE/'evidence_manifest.json').write_text(json.dumps({'status':'generated from frozen result artifacts',
        'not_new_timings':False,'error_metric':'unweighted relative L2 of stored arrays, percent',
        'results':results,'source_sha256':SOURCES},indent=2)+'\n')
    print(f'Generated {len(list(FIG.glob("*.pdf")))} figures and {len(list(TAB.glob("*.tex")))} tables.')


if __name__=='__main__': main()
