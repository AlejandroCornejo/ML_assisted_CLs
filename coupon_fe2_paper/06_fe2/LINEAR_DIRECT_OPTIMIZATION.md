# Local/batched optimizations for HPROM and D-HPROM-ANN

Date: 2026-09-06. Both drivers now default to `--implementation optimized`.
Their original implementations remain accessible with `--implementation baseline`.
The original result files are preserved.

## Numerical models and implementation changes

The linear HPROM retains its 39 POD modes, fixed residual ECM rule on the
135-element MDPA, and separate fixed stress rule on the 73-element MDPA
(183 distinct physical elements). `linear_hprom_fast.py` projects local
element matrices/forces directly into the 39-dimensional system. It evaluates
the six stress perturbations of the same IFT tangent together. The periodic
lifting derivative is evaluated only at the residual mesh DOFs. Newton,
continuation/substeps, and tolerances are unchanged; no ANN enters this tier.

The D-HPROM-ANN retains the direct E-to-slave-amplitude closure and the adaptive
MAW stress rule on exactly 10 MDPA elements. `direct_hprom_ann_fast.py` batches
the seven evaluations of the existing central-difference tangent, across both
stencil states and macro points in each worker chunk. It only computes decoder
values, since decoder Jacobians are unused by this stress finite difference.
Batching is capped at 128 macro points per group to bound temporary memory.
No microscopic equilibrium solve is introduced or removed: this model was
already direct.

`reduced_stress_batch.py` supplies the shared batched kinematics and stress
integration, using geometry, quadrature, materials and equation maps extracted
from the actual reduced MDPA assemblers. It uses the established microscopic
Neo-Hookean constitutive function and averages first-Piola stress before the
macro second-Piola conversion. No full-RVE online element assembly is used.

## Verification

`verify_linear_direct_fast.py` checks seven heterogeneous states, including
zero, small strain, and points from the actual coupon solution. It compares
stress, tangent and converged coordinates against the original implementations,
and independently finite-differences stress with a different perturbation
size. In the linear model these derivative checks re-solve micro equilibrium.
An additional 131-point direct batch tests the memory-cap split branch.

| Maximum normalized error | Linear HPROM | D-HPROM-ANN |
|---|---:|---:|
| Stress vs original | 4.71e-14 | 4.20e-13 |
| Tangent vs original | 2.68e-10 | 7.25e-10 |
| Converged coordinates vs original | 1.23e-15 | 0 |
| Tangent vs independent stress FD | 3.51e-9 | 2.38e-9 |

The extra direct batch agrees with the original at relative stress error
6.38e-15 and tangent error 5.10e-10. Complete checks and isolated constitutive
timing samples are saved in `linear_direct_fast_verification.json`.

## Complete FE2 benchmark protocol

`benchmark_linear_direct_optimized.sh` runs one original-implementation recheck
and three optimized repeats for each model, sequentially, with 20 workers and
one numerical-library thread per worker. Each is a fresh complete simulation
with the same macro mesh, loads, 20 increments and convergence tolerances.
The wall timer includes macro solution, microscopic stress/tangent evaluations,
communication and lazy worker construction. Parent setup and result export are
excluded, matching the established FE2 driver timing scope.

`compare_linear_direct_optimized.py` checks mesh hashes, force, worker count,
ECM sizes, convergence, coverage, material-call count and the entire saved
strain path. It preserves every timing and reports medians of all three
optimized repeats in `linear_direct_optimization_comparison.json`.
The error metric is the unweighted relative L2 norm of stored nodal/GP arrays.

## Final measured results

| Model | Original saved time [s] | Original recheck [s] | Optimized repeats [s] | Optimized median [s] | Speedup vs saved FOM |
|---|---:|---:|---|---:|---:|
| Linear HPROM | 116.9620 | 120.2201 | 63.3535, 67.3484, 68.9530 | 67.3484 | 63.38x |
| D-HPROM-ANN | 11.9599 | 14.6411 | 3.7880, 3.7585, 3.9695 | 3.7880 | 1126.90x |

The optimized linear HPROM is 1.74x faster than its saved original timing
(1.79x against the fresh recheck). The optimized D-HPROM-ANN is 3.16x faster
than its saved original timing (3.87x against the fresh recheck). Timing
variability is preserved in the repeats instead of selecting the fastest run.
The FOM reference is the previously saved 4268.6793 s run; it was not rerun here.

The previously optimized iterative HPROM-ANN median remains 30.1429 s, so
it is now 2.23x faster than the **also optimized** linear HPROM. The previous
3.88x comparison was against the unoptimized linear implementation and should
not be used for an all-optimized comparison.

All eight new complete simulations converged in four macro iterations per
increment, with all 2070 Gauss points inside the training box at every saved
step, and unchanged material-call counts. The fresh baseline fields are
identical to the original saved arrays. For optimized runs, maximum relative
L2 differences vs each original are:

| Field | Linear HPROM | D-HPROM-ANN |
|---|---:|---:|
| Displacement | 5.15e-16 | 3.85e-16 |
| Final strain | 1.68e-14 | 1.66e-14 |
| Final stress | 1.78e-14 | 1.78e-14 |
| Complete saved strain path | 1.66e-14 | 1.63e-14 |

Errors against FOM remain unchanged to the shown precision:

| Model | u error [%] | E error [%] | S error [%] |
|---|---:|---:|---:|
| Linear HPROM | 0.0056504 | 0.0165286 | 0.0009869 |
| D-HPROM-ANN | 0.0232407 | 0.0485112 | 0.0096697 |

Median-time result files are
`hprom_fe2_clean_timing_ecm_w20_f100kn_optimized_r2.{json,npz,log}` and
`dhprom_ann_fe2_clean_timing_maw10_direct_w20_f100kn_optimized_r1.{json,npz,log}`.
All repeats and the original implementations remain available for auditing.

## Reproduction

From the project root, use the existing NumPy/SciPy and Kratos environment:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/coupon_fe2_mpl
export PYTHONPATH=coupon_fe2_paper/.pydeps:/home/sares/Kratos_Eigen_Check/bin/Release
python3 -B coupon_fe2_paper/06_fe2/run_hprom_fe2.py \
  --implementation optimized --workers 20 --tag new_linear_optimized
python3 -B coupon_fe2_paper/06_fe2/run_hprom_ann_fe2.py \
  --direct --implementation optimized --workers 20 --tag new_direct_optimized
```

Use new tags to preserve previous outputs. The benchmark shell script refuses
to overwrite its existing named results.
