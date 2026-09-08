"""Compare local/batched iterative law against independent sparse baseline.

Checks off-equilibrium residuals (including pinned DOF lifting), decoded
stresses, converged coordinates, IFT tangents and re-solved stress derivatives.
Benchmarks exclude construction and profiling overhead; FE2 wall timing is
measured separately by run_hprom_ann_fe2.py.
"""
import contextlib
import io
import json
from pathlib import Path
import time

import numpy as np

from maw_hprom_ann_law import MAWHPROMANN
from maw_hprom_ann_fast import FastMAWHPROMANN

HERE = Path(__file__).resolve().parent


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        base = MAWHPROMANN('/tmp/coupon_maw_verify_baseline')
        fast = FastMAWHPROMANN('/tmp/coupon_maw_verify_local')
    path = np.load(HERE / 'hprom_ann_fe2_clean_timing_maw10_w20_f100kn.npz')['E_final']
    states = np.vstack((np.zeros(3), [.12, -.05, -.04], path[[0, 333, 1000, 1700, 2069]]))
    errors = dict(residual=0., fixed_state_stress=0., q=0., stress=0., tangent=0., resolved_fd=0.)
    def error(key, actual, expected, floor=1.):
        value = float(np.linalg.norm(actual-expected) / max(np.linalg.norm(expected), floor))
        errors[key] = max(errors[key], value)
    for E in states:
        # Nonzero residual ensures a genuinely relative assembly comparison.
        q_trial = .93*E + np.array([.001, -.001, .0005])
        error('residual', fast._residual_state(E, q_trial), base._residual_state(E, q_trial))
        error('fixed_state_stress', fast._stress_from_state(E, q_trial), base._stress_from_state(E, q_trial))
        S0, C0, q0 = base.stress_and_tangent(E, return_state=True)
        S1, C1, q1 = fast.stress_and_tangent(E, return_state=True)
        error('q', q1, q0)
        error('stress', S1, S0)
        error('tangent', C1, C0)
        # Independent FD differentiates re-solved equilibria, not fixed q.
        h = 3e-6
        for k in range(3):
            de = np.eye(3)[k]*h
            qp = fast.solve(E+de, q_init=q1, E_start=E)
            qm = fast.solve(E-de, q_init=q1, E_start=E)
            fd = (fast._stress_from_state(E+de, qp)-fast._stress_from_state(E-de, qm))/(2*h)
            error('resolved_fd', C1[:, k], fd)
    assert errors['residual'] < 1e-10, errors
    assert errors['fixed_state_stress'] < 1e-10, errors
    assert errors['q'] < 1e-9, errors
    assert errors['stress'] < 1e-8, errors
    assert errors['tangent'] < 1e-6, errors
    assert errors['resolved_fd'] < 1e-5, errors
    E = np.array([.12, -.05, -.04])
    target = E + np.array([.005, -.002, -.002])
    q = base.solve(E)
    timing = {}
    for name, law in [('baseline', base), ('optimized', fast)]:
        law.stress_and_tangent(target, q_init=q, E_start=E)
        samples = []
        for repeat in range(3):
            tic = time.perf_counter()
            for _ in range(40):
                law.stress_and_tangent(target, q_init=q, E_start=E)
            samples.append((time.perf_counter()-tic)/40)
        timing[name] = samples
    result = dict(status='PASS', states=states.tolist(), max_errors=errors,
                  seconds_per_response=timing,
                  micro_speedup=float(np.median(timing['baseline'])/np.median(timing['optimized'])))
    (HERE/'maw_hprom_ann_fast_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('VERIFICATION', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
