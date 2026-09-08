"""Validate complete FE2 fields and summarize sequential optimization timings."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = 'hprom_ann_fe2_clean_timing_maw10_w20_f100kn'


def main():
    names = dict(
        fom='fom_fe2_clean_timing_fomfe2_w4_f100kn',
        hprom='hprom_fe2_clean_timing_ecm_w20_f100kn',
        original=PREFIX,
        baseline_recheck=PREFIX+'_baseline_recheck',
        optimized_r1=PREFIX+'_optimized_r1',
        optimized_r2=PREFIX+'_optimized_r2',
        optimized_r3=PREFIX+'_optimized_r3',
    )
    summaries = {k: json.loads((HERE/(v+'.json')).read_text()) for k, v in names.items()}
    arrays = {k: np.load(HERE/(v+'.npz')) for k, v in names.items()}
    rows = {}
    for name in names:
        meta, data = summaries[name], arrays[name]
        assert meta['status'] == 'converged' and meta['n_steps_completed'] == 20
        for key in ['coords', 'connectivity']:
            assert np.array_equal(data[key], arrays['fom'][key]), (name, key)
        for key in ['macro_mesh_sha256', 'rve_mesh_sha256', 'force_per_end']:
            assert meta[key] == summaries['fom'][key], (name, key)
        errors = {}
        for field in ['u_nodal', 'E_final', 'S_final']:
            ref = arrays['fom'][field]
            errors[field] = float(np.linalg.norm(data[field]-ref)/np.linalg.norm(ref))
        rows[name] = dict(seconds=meta['wall_seconds'],
                          speedup_vs_fom=summaries['fom']['wall_seconds']/meta['wall_seconds'],
                          relative_l2_vs_fom=errors)
        if name.startswith('optimized') or name == 'baseline_recheck':
            same = {}
            for field in ['u_nodal', 'E_final', 'S_final', 'E_path']:
                ref = arrays['original'][field]
                same[field] = float(np.linalg.norm(data[field]-ref)/np.linalg.norm(ref))
                assert same[field] < 1e-7, (name, field, same[field])
            rows[name]['relative_l2_vs_original'] = same
            assert [s['iterations'] for s in meta['macro_newton']] == [4]*20
            assert all(s['coverage']['outside'] == 0 for s in meta['macro_newton'])
    opt = [rows[k]['seconds'] for k in ['optimized_r1', 'optimized_r2', 'optimized_r3']]
    result = dict(status='PASS', field_metric='unweighted relative L2 of stored nodal/GP arrays',
                  runs=rows, optimized_seconds_range=[min(opt), max(opt)],
                  optimized_seconds_median=float(np.median(opt)),
                  speedup_vs_rechecked_baseline=rows['baseline_recheck']['seconds']/float(np.median(opt)),
                  speedup_vs_linear_hprom=rows['hprom']['seconds']/float(np.median(opt)))
    (HERE/'maw_hprom_ann_optimization_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
