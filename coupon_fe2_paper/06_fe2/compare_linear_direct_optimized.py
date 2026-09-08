"""Audit the optimized linear/direct FE2 runs and preserve all repeat timings."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def load(stem):
    return json.loads((HERE/(stem+'.json')).read_text()), np.load(HERE/(stem+'.npz'))


def main():
    fm, fd = load('fom_fe2_clean_timing_fomfe2_w4_f100kn')
    stems = dict(linear='hprom_fe2_clean_timing_ecm_w20_f100kn',
                 direct='dhprom_ann_fe2_clean_timing_maw10_direct_w20_f100kn')
    report = dict(status='PASS', metric='unweighted relative L2 of stored nodal/GP arrays',
                  fom_seconds=fm['wall_seconds'], models={})
    for model, stem in stems.items():
        original_meta, original_data = load(stem)
        rows = {}
        for run in ['baseline_recheck','optimized_r1','optimized_r2','optimized_r3']:
            meta, data = load(stem+'_'+run)
            assert meta['status'] == 'converged' and meta['n_steps_completed'] == 20
            for key in ['coords','connectivity']:
                assert np.array_equal(data[key],fd[key]), (model,run,key)
            for key in ['macro_mesh_sha256','rve_mesh_sha256','force_per_end','workers']:
                assert meta[key] == original_meta[key] == fm[key], (model,run,key)
            assert [s['iterations'] for s in meta['macro_newton']] == [4]*20
            assert all(s['coverage']['outside'] == 0 for s in meta['macro_newton'])
            assert meta['material_calls'] == original_meta['material_calls']
            if model == 'linear':
                assert meta['hprom_modes'] == 39
                assert meta['residual_ecm_elements'] == 135 and meta['stress_ecm_elements'] == 73
            else:
                assert meta['residual_ecm_elements'] == 0 and meta['stress_ecm_elements'] == 10
                supports=[m['stress_support_full_indices'] for m in meta['worker_ecm_mdpa']]
                assert all(s == original_meta['worker_ecm_mdpa'][0]['stress_support_full_indices'] for s in supports)
                assert min(m['minimum_adaptive_weight'] for m in meta['worker_ecm_mdpa']) > 0
            err_fom, err_orig = {}, {}
            for field in ['u_nodal','E_final','S_final','E_path']:
                value = data[field]
                err_orig[field] = float(np.linalg.norm(value-original_data[field])/np.linalg.norm(original_data[field]))
                assert err_orig[field] < 1e-7, (model,run,field,err_orig[field])
                if field != 'E_path':
                    err_fom[field] = float(np.linalg.norm(value-fd[field])/np.linalg.norm(fd[field]))
            rows[run]=dict(seconds=meta['wall_seconds'],relative_l2_vs_original=err_orig,
                           relative_l2_vs_fom=err_fom)
        timings=[rows[k]['seconds'] for k in ['optimized_r1','optimized_r2','optimized_r3']]
        med=float(np.median(timings))
        report['models'][model]=dict(original_seconds=original_meta['wall_seconds'],
            runs=rows,optimized_seconds_median=med,optimized_seconds_range=[min(timings),max(timings)],
            speedup_vs_original=original_meta['wall_seconds']/med,
            speedup_vs_rechecked_baseline=rows['baseline_recheck']['seconds']/med,
            speedup_vs_fom=fm['wall_seconds']/med)
    previous=json.loads((HERE/'maw_hprom_ann_optimization_comparison.json').read_text())
    report['iterative_ann_seconds_median']=previous['optimized_seconds_median']
    report['linear_over_iterative_ann_time']=report['models']['linear']['optimized_seconds_median']/previous['optimized_seconds_median']
    (HERE/'linear_direct_optimization_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
