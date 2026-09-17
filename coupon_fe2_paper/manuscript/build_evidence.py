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
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

FIG = HERE / 'figures'
TAB = HERE / 'tables'
SOURCES = {}


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


def finite_element_results():
    path = ROOT / '06_fe2'
    cases = [
        ('FOM', 'fom_fe2_clean_timing_fomfe2_w4_f100kn', 1),
        ('HPROM', 'hprom_fe2_clean_timing_ecm_w20_f100kn_optimized_r', 3),
        ('HPROM--ANN', 'hprom_ann_fe2_clean_timing_maw10_w20_f100kn_optimized_r', 3),
        ('D-HPROM--ANN', 'dhprom_ann_fe2_clean_timing_maw10_direct_w20_f100kn_optimized_r', 3),
        ('ICNN', 'pann_fe2_clean_timing_icnn32_t4_f100kn', 1),
        ('ICKAN', 'pann_fe2_clean_timing_ickan32_t8_f100kn', 1),
        ('Free', 'pann_fe2_clean_timing_free_t12_f100kn', 1),
        ('Regression', 'pann_fe2_clean_timing_regression_t16_f100kn', 1),
    ]
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
    rows = []
    for row in results:
        errors = row['errors_percent']
        parallel = f"{row['concurrency']}" + ('p' if row['concurrency_kind']=='processes' else 't')
        rows.append([row['model'], parallel, str(row['repeats']), f"{row['seconds']:.3f}",
                     f"{row['speedup']:.1f}", *(f'{errors[k]:.5f}' for k in errors)])
    table('fe2_comparison.tex', ['Model', 'CPU', '$n_{\\rm run}$', '$t$ [s]', '$t_0/t$',
          '$e_u$ [\\%]', '$e_E$ [\\%]', '$e_S$ [\\%]'], rows, 'lrrrrrrr',
          'p: worker processes, one numerical-library thread each; t: PyTorch threads. '
          'For $n_{\\rm run}=3$, time is the median; for $n_{\\rm run}=1$, it is the archived single run. '
          'These are online wall times, not offline-inclusive or matched-thread speedups.')
    table('timing_repeats.tex', ['Model', '$t_1$ [s]', '$t_2$ [s]', '$t_3$ [s]'],
          [[r['model'], *(f'{t:.3f}' for t in r['times'])] for r in results if r['repeats']==3], 'lrrr')
    fig, ax = plt.subplots(figsize=(6.9, 3.8), layout='constrained')
    for i, row in enumerate(results[1:]):
        x, y = row['seconds'], row['errors_percent']['u_nodal']
        ax.scatter(x, y, s=44, color=COLORS[i], zorder=3)
        if len(row['times']) > 1:
            ax.plot([min(row['times']), max(row['times'])], [y,y], color=COLORS[i], lw=2)
        ax.annotate(row['model'].replace('--','–'), (x,y), xytext=(5,5), textcoords='offset points', fontsize=8)
    ax.set(xscale='log', yscale='log', xlabel='Online wall time [s]', ylabel='Final displacement error [%]',
           xlim=(.7,150), ylim=(.003,1.6))
    ax.grid(alpha=.2, which='both')
    save(fig, 'cost_accuracy')
    field_figures(reference)
    return results


def field_figures(data):
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
    fields = [(1000*np.linalg.norm(data['u_nodal'],axis=1), r'$\|u\|$ [mm]', 'viridis'),
              (recover(data['E_final'][:,0]), r'$E_{11}$', 'magma'),
              (recover(data['S_final'][:,0]/1e6), r'$S_{11}$ [MPa]', 'viridis'),
              (recover(data['S_final'][:,1]/1e6), r'$S_{22}$ [MPa]', 'RdBu_r'),
              (recover(data['S_final'][:,2]/1e6), r'$S_{12}$ [MPa]', 'RdBu_r')]
    fig, axes = plt.subplots(5,1,figsize=(7.1,6.3),layout='constrained')
    for i, (ax,(val,label,cmap)) in enumerate(zip(axes,fields)):
        limits = dict(vmin=-max(abs(val)),vmax=max(abs(val))) if i>=3 else {}
        im = ax.tripcolor(tri,val,shading='gouraud',cmap=cmap,**limits)
        ax.triplot(coarse,color='white',lw=.15,alpha=.4)
        ax.set_aspect('equal')
        ax.set_ylabel('$y$ [mm]')
        if i<4: ax.tick_params(labelbottom=False)
        else: ax.set_xlabel('$x$ [mm]')
        fig.colorbar(im,ax=ax,fraction=.023,pad=.015).set_label(label)
    save(fig,'coupon_fields')


def constitutive_results():
    audit = read_json(ROOT/'06_pann/enrichment_results/selected_models_audit.json')
    rows, fig = [], plt.figure(figsize=(6.9,3.4),layout='constrained')
    ax = fig.subplots()
    for name, token, color in [('ICNN','flex_icnn',COLORS[3]), ('ICKAN','flex_ickan',COLORS[4])]:
        entry = next(v for k,v in audit.items() if token in k)
        m = entry['metrics']
        rows.append([name, f"{100*m['test']['stress']:.4f}", f"{100*m['test']['energy']:.4f}",
                     f"{100*m['probe']['stress']:.4f}",
                     f"{100*m['test']['stress_sample_relative_percentiles'][-1]:.2f}",
                     f"{100*m['probe']['stress_sample_relative_percentiles'][-1]:.2f}"])
        rings = entry['probe_by_ring']
        ax.plot([float(r) for r in rings],[100*rings[r]['stress'] for r in rings], 'o-', label=name,color=color)
    # Keep out-of-domain evidence in the supplement; do not restore probe
    # columns in the main-paper test table when regenerating artifacts.
    test_rows = [[row[0], row[1], row[2], row[4]] for row in rows]
    table('constitutive_errors.tex',
          ['Model', '$e_S^{\\rm test}$', '$e_W^{\\rm test}$', 'Worst test'],
          test_rows, 'lrrr',
          'All values are percentages. Stress and energy errors are aggregate array norms; '
          'the last column is the maximum per-state stress error. Material A: 400 independent test states.')
    table('constitutive_probe_errors.tex',['Model', '$e_S^{\\rm test}$', '$e_W^{\\rm test}$', '$e_S^{\\rm probe}$',
          'Worst test', 'Worst probe'],rows,'lrrrrr',
          'All values are percentages. The last two columns are maximum per-state stress errors; '
          'the first three are aggregate array norms. Test: 400 states; probe: 345 finite reference states out of 350.')
    ax.set(xlabel='Sampling-box overshoot factor (not a load factor)',ylabel='Probe stress error [%]')
    ax.legend(frameon=False); ax.grid(alpha=.2)
    save(fig,'probe_errors')


def mechanics():
    cycle=read_json(ROOT/'06_pann/mechanics_witness_results/current_rve_cycle_audit.json')['cycle']
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
    fig, (left,right)=plt.subplots(1,2,figsize=(8.0,3.8),layout='constrained',gridspec_kw={'width_ratios':[1,1.6]})
    center=cycle['centre_E']; h=cycle['half_width_in_E11_and_E22']
    x=np.array([-1,1,1,-1,-1])*h+center[0]
    y=np.array([-1,-1,1,1,-1])*h+center[1]
    left.plot(x,y,'o-',color='#334155',ms=3)
    left.annotate('',xy=(center[0]+h/2,center[1]-h),xytext=(center[0]-h/2,center[1]-h),arrowprops={'arrowstyle':'->'})
    left.set(xlabel='$E_{11}$',ylabel='$E_{22}$',title=rf'$\gamma_{{12}}={center[2]:.5f}$')
    left.ticklabel_format(axis='both',style='plain',useOffset=False)
    left.tick_params(axis='x',rotation=35)
    for name,color in [('Regression',COLORS[6]),('Free',COLORS[5]),('ICNN',COLORS[3]),('ICKAN',COLORS[4])]:
        values=[max(abs(conv[q][name]),1e-12) for q in conv]
        right.loglog([int(q) for q in conv],values,'o-',label=name,color=color,ms=3)
    for name, color, marker in [('HPROM',COLORS[0],'v'), ('HPROM--ANN',COLORS[2],'s'),
                                ('D-HPROM--ANN',COLORS[1],'^')]:
        right.loglog([int(q) for q in conv], [abs(conv[q][name]) for q in conv],
                     marker+'--', label=name.replace('--','–'), color=color, ms=3, lw=1.2)
    right.scatter([8],[abs(cycle['FOM']['stress_work_J_per_m3'])],marker='x',color='black',label='FOM (8/edge)',zorder=4)
    right.set(xlabel='Gauss points per cycle edge',ylabel=r'$|\oint s\cdot de|$ [J m$^{-3}$]')
    right.legend(frameon=False,fontsize=7,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.24),
                 columnspacing=.9,handlelength=1.6); right.grid(alpha=.2)
    save(fig,'cycle_convergence')
    names=('Regression','Free','ICNN','ICKAN','HPROM','HPROM--ANN','D-HPROM--ANN')
    rows=[[name, *(f'{conv[q][name]:.6g}' for q in conv)] for name in names]
    table('cycle_convergence.tex',['Model',*conv.keys()],rows,'lrrrrr',
          'Signed cycle work in J m$^{-3}$; column headings give Gauss points per edge. '
          'FOM at 8 points per edge: '
          f"{cycle['FOM']['stress_work_J_per_m3']:.6g}"+' J m$^{-3}$. '
          'The spline quadrature error decreases with refinement; it is not physical dissipation. '
          'Reversal and equilibrium-tolerance controls are discussed in the text.')
    audit=read_json(ROOT/'06_pann/mechanics_witness_results/current_rve_mechanics_witnesses.json')
    rows=[]
    for key,label in [('held_out_test','Held-out test'),('converged_probe','Finite-label probe'),('uniform_training_box','In-box audit (unlabelled)')]:
        cloud=audit['rank_one']['clouds'][key]
        rows.append([label,str(cloud['models']['Free']['n_states']), *(f"{cloud['models'][n]['curvature_Pa']/1e6:.3f}" for n in ('Free','ICNN','ICKAN'))])
        assert all(v['n_b_directions']==180 for v in cloud['models'].values())
    table('rank_one.tex',['State set','$N$','Free','ICNN','ICKAN'],rows,'lrrrr',
          'Minimum sampled rank-one curvature in MPa. Minima need not occur at the same state or direction. '
          'The additional in-box cloud has no independently computed FOM labels; '
          '180 unit directions for $\\bm b$ are sampled, with minimization over $\\bm a$ via an eigenproblem. '
          'None of these finite samples is a global certificate.')


def geometry():
    sys.path.insert(0,str(ROOT))
    import config as cfg
    source(ROOT/'config.py')
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
    fig,(left,right)=plt.subplots(1,2,figsize=(7,3.1),layout='constrained')
    left.triplot(xy[:,0],xy[:,1],t[:,:3],color='#687c90',lw=.22)
    left.set(aspect='equal',xlabel='$X_1/\\ell$',ylabel='$X_2/\\ell$',title='Periodic cell: 1546 T6 elements')
    left.set_xticks([-1,0,1],['−1/2','0','1/2']);left.set_yticks([-1,0,1],['−1/2','0','1/2'])
    a,b=cfg.ellipse_semi_axes()
    right.add_patch(plt.Rectangle((-1,-1),2,2,facecolor='#dae3eb',edgecolor='#334155'))
    right.add_patch(Ellipse((0,0),2*a,2*b,angle=cfg.ELLIPSE_ANGLE_DEG,facecolor='white',edgecolor='#334155'))
    th=np.deg2rad(cfg.ELLIPSE_ANGLE_DEG)
    right.plot([0,a*np.cos(th)],[0,a*np.sin(th)],'--',color='#bd5d3b',lw=1)
    right.text(.35,.02,r'$30^\circ$',color='#bd5d3b')
    right.text(0,-1.25,'Void fraction 0.20; aspect ratio 2:1',ha='center',fontsize=8)
    right.set(xlim=(-1.15,1.15),ylim=(-1.4,1.15),aspect='equal',title='Geometry, not an imposed material symmetry')
    right.axis('off')
    save(fig,'rve_geometry')
    supports_figure(xy,t)


def supports_figure(xy,t):
    folder=ROOT/'05_validation'
    fixed=np.load(source(folder/'ecm_supports.npz'))
    adaptive_r=np.load(source(folder/'maw_res_long10.npz'))['res_10_z'].astype(int)
    adaptive_s=np.load(source(folder/'maw_phase2_sig.npz'))['sig_10_z'].astype(int)
    cases=[('HPROM',fixed['z_res'].astype(int),fixed['z_sig'].astype(int),183),
           ('HPROM–ANN',adaptive_r,adaptive_s,19),
           ('D-HPROM–ANN',np.array([],dtype=int),adaptive_s,10)]
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


COLORS=['#315c80','#d58b28','#7d4b2d','#238268','#7754a0','#738ca1','#c44939']


def main():
    FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
    source(HERE/'build_evidence.py')
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9,
                         'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,
                         'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    results=finite_element_results()
    constitutive_results(); mechanics(); geometry()
    (HERE/'evidence_manifest.json').write_text(json.dumps({'status':'generated from archived artifacts',
        'not_new_timings':True,'error_metric':'unweighted relative L2 of stored arrays, percent',
        'results':results,'source_sha256':SOURCES},indent=2)+'\n')
    print(f'Generated {len(list(FIG.glob("*.pdf")))} figures and {len(list(TAB.glob("*.tex")))} tables.')


if __name__=='__main__': main()
