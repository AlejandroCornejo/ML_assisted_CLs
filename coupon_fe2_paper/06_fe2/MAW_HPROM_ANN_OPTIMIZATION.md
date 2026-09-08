# Iterative MAW-HPROM-ANN FE2 optimization — 2026-09-06

The iterative model now runs in approximately 30 seconds on this coupon,
with unchanged fields. The default of `run_hprom_ann_fe2.py` is now
`--implementation optimized`; `--implementation baseline` preserves the
original implementation. `--direct` still selects the separate direct model.

## What changed

A representative continued micro response spent about 64% of its profiled
time in the macro IFT tangent. Its 12 residual and 12 stress perturbation
evaluations each constructed a sparse global microscopic stiffness, even
though those evaluations only require residual or stress values.

`maw_hprom_ann_fast.py` subclasses the unchanged reference law and:

- Projects local element stiffness/force arrays directly into three coordinates
  during microscopic modified Newton, avoiding global sparse assembly.
- Evaluates the same central-difference IFT stencil in batches: one residual
  batch and one stress batch, with no unused stiffness assembly.
- Computes the periodic lifting once per micro substep and stores its geometric
  arm vectors for batched evaluations.

The decoder/checkpoints, adaptive weight fields, actual 10+10 element MDPA
supports (19 distinct physical elements), micro Newton tolerance `1e-10`,
strain substep rule, macro tolerances, and 20 load increments are unchanged.
The model still solves microscopic equilibrium. The IFT tangent still includes
decoder curvature and adaptive-weight dependence through differentiation of
the actual residual. No retraining, reduced accuracy or direct closure is used.

## Verification

`verify_maw_hprom_ann_fast.py` checks seven states, including zero strain and
states from the actual FE2 final field, against the independent sparse law.
Maximum differences:

| Quantity | Normalized difference |
|---|---:|
| Off-equilibrium residual | 2.23e-15 |
| Fixed-state homogenized stress | 1.29e-14 |
| Converged stress | 5.90e-15 |
| Macro tangent vs original IFT | 6.94e-10 |
| Macro tangent vs independently re-solved stress FD | 1.57e-7 |

Complete FE2 comparisons pass as well: maximum relative L2 difference vs the
original saved solution is 4.32e-16 for displacement, 1.68e-14 for final strain,
1.77e-14 for final stress, and 1.65e-14 for the entire saved strain path.
Every optimized run has 4 macro Newton iterations in each of 20 increments
and 2070/2070 in-box Gauss points at every converged step.

Errors against FOM remain: u = 0.0519962%, E = 0.1496790%, S = 0.0372712%.
These are unweighted relative L2 errors on stored nodal/GP arrays, not maxima.

## Timing protocol and results

All new timing runs are complete, fresh simulations with 20 worker processes,
one numerical-library thread per worker, CPUQuota=2000%, no overlapping
benchmark jobs, and output serialization outside the timer. The wall timer
includes macro assembly/solution, constitutive stress and tangent, communication,
and lazy worker model construction. Parent mesh/setup is excluded, as in the
previous runner. This clarifies the original driver's overly broad claim that
all setup was excluded. Scalar profiling/verification runs were separate.

| Run | Wall seconds | Source |
|---|---:|---|
| Original HPROM-ANN | 102.7664 | previous saved reference |
| Optimized repeat 1 | 27.9473 | new complete run |
| Original implementation recheck | 106.7782 | new complete run |
| Optimized repeat 2 | 34.6280 | new complete run |
| Optimized repeat 3 | 30.1429 | new complete run |

Use the median of all three optimized runs: **30.1429 s**, with observed range
27.9473–34.6280 s. This is **3.54x** faster than the original implementation
recheck, **3.88x** faster than the saved linear HPROM time (116.9620 s), and
approximately **141.6x** faster than the saved FOM time (4268.6793 s).
Subsequent optimization of the linear and direct tiers is documented in
`LINEAR_DIRECT_OPTIMIZATION.md`. The all-optimized linear HPROM now takes
67.3484 s (median), so the iterative ANN advantage relative to that version
is 2.23x. The 3.88x figure above describes the earlier unoptimized linear code.
The range is retained explicitly; the fastest repeat alone is not the headline.
The FOM and linear HPROM timings were not remeasured in this optimization pass.

## Artifacts and reproduction

- `maw_hprom_ann_fast_verification.json`: independent constitutive checks.
- `maw_hprom_ann_optimization_comparison.json`: all timings and field errors.
- `hprom_ann_fe2_clean_timing_maw10_w20_f100kn_optimized_r{1,2,3}.{json,npz,log}`:
  full optimized results; the median-time run is r3.
- `hprom_ann_fe2_clean_timing_maw10_w20_f100kn_baseline_recheck.{json,npz,log}`:
  new original-implementation timing, identical fields to the previous result.

From the project root, with the existing Kratos runtime and NumPy/SciPy paths:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/coupon_fe2_mpl \
PYTHONPATH=coupon_fe2_paper/.pydeps:/home/sares/Kratos_Eigen_Check/bin/Release \
python3 -B coupon_fe2_paper/06_fe2/run_hprom_ann_fe2.py \
  --implementation optimized --workers 20 --tag new_optimized_run
```

`benchmark_maw_optimized.sh` records the sequential benchmark commands and
refuses to overwrite existing results. `compare_maw_optimized.py` recomputes
the field comparisons and the timing summary from the saved runs.
