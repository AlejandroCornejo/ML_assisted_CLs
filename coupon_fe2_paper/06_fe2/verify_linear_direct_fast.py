"""Independent original/optimized constitutive checks for both remaining tiers."""
import contextlib
import io
import json
from pathlib import Path
import time
import numpy as np

import run_hprom_fe2  # installs the project's established import paths
from linear_hprom_ecm import LinearHPROMECM
from linear_hprom_fast import FastLinearHPROMECM
from direct_hprom_ann_law import MAWDHPROMANN
from direct_hprom_ann_fast import FastMAWDHPROMANN

HERE = Path(__file__).resolve().parent


def rel(actual, ref, floor=1.):
    return float(np.linalg.norm(actual-ref)/max(floor, np.linalg.norm(ref)))


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        linear = LinearHPROMECM('/tmp/coupon_check_linear_base')
        linear_fast = FastLinearHPROMECM('/tmp/coupon_check_linear_fast')
        direct = MAWDHPROMANN('/tmp/coupon_check_direct_base')
        direct_fast = FastMAWDHPROMANN('/tmp/coupon_check_direct_fast')
    cloud = np.load(HERE/'fom_fe2_clean_timing_fomfe2_w4_f100kn.npz')['E_final']
    states = np.vstack((np.zeros(3), [1e-4, -4e-5, -3e-5], [.12, -.05, -.04], cloud[[0,333,1000,2069]]))
    report = {}
    for name, base, fast in [('linear',linear,linear_fast), ('direct',direct,direct_fast)]:
        errs = dict(stress=0., tangent=0., q=0., resolved_fd=0.)
        for E in states:
            S0,C0,q0 = base.stress_and_tangent(E,return_state=True)
            S,C,q = fast.stress_and_tangent(E,return_state=True)
            for key, actual, ref in [('stress',S,S0), ('tangent',C,C0), ('q',q,q0)]:
                errs[key] = max(errs[key], rel(actual,ref))
            # Independent centered derivative, with a different step size;
            # the linear case re-solves equilibrium in each perturbed state.
            for j in range(3):
                dE = np.eye(3)[j]*3e-6
                if name == 'linear':
                    qp = fast.solve(E+dE,q_init=q,E_start=E)[0]
                    qm = fast.solve(E-dE,q_init=q,E_start=E)[0]
                    Sp,Sm = fast._stress_from_state(E+dE,qp),fast._stress_from_state(E-dE,qm)
                else:
                    Sp,Sm = base.evaluate_stress(E+dE),base.evaluate_stress(E-dE)
                errs['resolved_fd'] = max(errs['resolved_fd'],rel(C[:,j],(Sp-Sm)/6e-6))
        assert errs['stress'] < 1e-8 and errs['q'] < 1e-9, errs
        assert errs['tangent'] < 1e-6 and errs['resolved_fd'] < 1e-5, errs
        E=np.array([.12,-.05,-.04]); target=E+np.array([.005,-.002,-.002])
        q=base.stress_and_tangent(E,return_state=True)[2]
        times={}
        for label, law in [('baseline',base),('optimized_scalar',fast)]:
            law.stress_and_tangent(target,q_init=q,E_start=E)
            samples=[]
            for _ in range(3):
                tic=time.perf_counter()
                for _ in range(40): law.stress_and_tangent(target,q_init=q,E_start=E)
                samples.append((time.perf_counter()-tic)/40)
            times[label]=samples
        if name == 'direct':
            Q = cloud[np.linspace(0,len(cloud)-1,131,dtype=int)]
            S,C,_=fast.stress_and_tangent_batch(Q)
            # Check heterogeneous batches and the memory-bound split branch.
            Sb,Cb=zip(*(base.stress_and_tangent(e) for e in Q))
            errs['batch_stress']=rel(S,np.array(Sb))
            errs['batch_tangent']=rel(C,np.array(Cb))
            assert errs['batch_stress'] < 1e-10 and errs['batch_tangent'] < 1e-6,errs
            times['optimized_batch']=[]
            for _ in range(3):
                tic=time.perf_counter()
                for _ in range(10): fast.stress_and_tangent_batch(Q[:104])
                times['optimized_batch'].append((time.perf_counter()-tic)/1040)
        report[name]=dict(max_errors=errs,seconds_per_response=times)
    report['status']='PASS'
    (HERE/'linear_direct_fast_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('VERIFICATION',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
