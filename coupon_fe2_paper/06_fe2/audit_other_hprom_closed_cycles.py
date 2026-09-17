"""Complete the frozen cycle comparison with affine HPROM and D-HPROM--ANN.

Reuses the existing cycle geometry; does not change production laws or timings.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile

# The existing audit configures the single-thread environment and import paths.
from audit_maw_closed_cycle import ROOT, HERE, digest, rectangle_points
import numpy as np
import linear_hprom_fast as linear_module
from direct_hprom_ann_fast import FastMAWDHPROMANN


def evaluate(law, centre, half_width, order, direct=False, reverse=False):
    points, increments = rectangle_points(centre, half_width, order)
    if reverse:
        points, increments = points[::-1], -increments[::-1]
    residuals = []
    if direct:
        stress = np.concatenate([law.evaluate_stress_batch(points[k:k+128])
                                 for k in range(0, len(points), 128)])
    else:
        q, previous, stresses = None, None, []
        for E in points:
            q, _, fint = law.solve(E, q_init=q, E_start=previous)
            stresses.append(law._stress_from_state(E, q))
            residuals.append(float(np.linalg.norm(law.Tflat.T @ fint.reshape(-1))))
            previous = E.copy()
        stress = np.asarray(stresses)
    assert np.isfinite(stress).all()
    work = np.einsum('ij,ij->i', stress, increments)
    return {
        'signed_work_J_per_m3': float(work.sum()),
        'absolute_accumulated_work_J_per_m3': float(np.abs(work).sum()),
        'max_stress_Pa': float(np.max(np.linalg.norm(stress, axis=1))),
        'max_reduced_residual_norm': max(residuals) if residuals else None,
        'gauss_queries': len(points),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orders', default='8,16,32,64,128')
    parser.add_argument('--output', type=Path, default=HERE/'other_hprom_closed_cycle_audit.json')
    args = parser.parse_args()
    orders = [int(x) for x in args.orders.split(',')]
    assert orders and min(orders) >= 2
    source = ROOT/'06_pann/mechanics_witness_results/current_rve_cycle_audit.json'
    cycle = json.loads(source.read_text())['cycle']
    centre, width = np.asarray(cycle['centre_E']), cycle['half_width_in_E11_and_E22']
    files = [source, ROOT/'04_training/decoder_basis_B_r39.npz', ROOT/'04_training/nslave.npz',
             ROOT/'05_validation/ecm_supports.npz', ROOT/'05_validation/maw_phase2_sig.npz',
             ROOT/'03_data/rve_mesh.mdpa', HERE/'linear_hprom_fast.py', HERE/'linear_hprom_ecm.py',
             HERE/'direct_hprom_ann_fast.py', HERE/'direct_hprom_ann_law.py',
             HERE/'reduced_stress_batch.py', HERE/'audit_maw_closed_cycle.py', Path(__file__)]
    result = {
        'scope': 'Frozen affine HPROM and D-HPROM--ANN closed cycles; not structural timing runs',
        'source_sha256': {str(p.relative_to(ROOT)):digest(p) for p in files},
        'centre_E': centre.tolist(), 'half_width_in_E11_and_E22': width,
        'orientation': 'counter-clockwise; reverse control independently evaluated',
        'micro_displacement_increment_tolerance': linear_module.NEWTON_TOL,
        'models': {},
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    with tempfile.TemporaryDirectory(prefix='coupon-other-hprom-cycles-') as work:
        for name, cls, direct in [('HPROM', linear_module.FastLinearHPROMECM, False),
                                  ('D-HPROM--ANN', FastMAWDHPROMANN, True)]:
            with contextlib.redirect_stdout(io.StringIO()):
                law = cls(Path(work)/name)
            record = {'convergence': {}}
            result['models'][name] = record
            for order in orders:
                row = evaluate(law, centre, width, order, direct=direct)
                record['convergence'][str(order)] = row
                save()
                print(name, 'order='+str(order), json.dumps(row), flush=True)
            record['reverse_control'] = evaluate(law, centre, width, max(orders), direct=direct, reverse=True)
            record['reverse_control']['order'] = max(orders)
            if not direct:
                original_tolerance = linear_module.NEWTON_TOL
                try:
                    linear_module.NEWTON_TOL = 1e-12
                    record['tight_tolerance_control'] = evaluate(law, centre, width, max(orders))
                    record['tight_tolerance_control'].update(order=max(orders), tolerance=1e-12)
                finally:
                    linear_module.NEWTON_TOL = original_tolerance
            record['status'] = 'completed'
            save()
            print(name, 'controls', json.dumps({k:v for k,v in record.items() if k!='convergence'}), flush=True)
        result['status'] = 'completed'
        save()
    print('OTHER_HPROM_CYCLES_COMPLETED', args.output, flush=True)


if __name__ == '__main__':
    main()
