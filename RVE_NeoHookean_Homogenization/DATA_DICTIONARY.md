# Data dictionary: strain/stress definitions in this repository

This RVE uses a **mixed boundary-value problem**: only a subset of boundary
nodes carry a prescribed (Dirichlet) affine displacement `u = (F-I)(X-Xc)`;
the rest of the boundary, and the material around the internal hole, is free.
Under this scheme, several different quantities can all reasonably be called
"the macroscopic stress" or "the macroscopic strain" for a given loading
step, and they are **not** required to agree with each other -- this is a
standard subtlety of computational homogenization under non-fully-periodic
boundary conditions (the classical Hill-Mandel volume-average/energy-
conjugate equivalence only holds exactly for specific boundary-condition
types). This repository genuinely contains more than one such quantity, and
conflating them silently is the single most consequential documentation gap
found while working on this project -- this file exists so it cannot recur.

## The three strain/stress quantities

| Name in this repo | What it actually is | Where it's computed |
|---|---|---|
| **Applied / nominal strain** | The macro strain `E` used to build the prescribed boundary displacement. Never itself "measured" from a solve. | `stage6_test_hprom.generate_safe_test_path` + `fom_solver_rve.BuildDynamicSegmentSteps` (the path), applied as a Dirichlet BC in `RunFomBatchSimulation`. |
| **Measured / "legacy" strain & stress** | The actual volume-average of the converged microscopic Green-Lagrange strain and 2nd-Piola stress fields, from a real nonlinear FE solve. | `fom_solver_rve.CalculateHomogenizedFromAssemblerWithElementWeights`, returned by `RunFomBatchSimulation` as `strain_hist`/`stress_hist`. This is what `hprom/ann/stage10_test_hprom_ann*.py` report today (`fom_strain.npy`/`fom_stress.npy`, etc.). |
| **"Direct" energy-conjugate strain, stress & energy** | `strain` = the *applied* strain (same as row 1). `energy` = the true converged microscopic Neo-Hookean energy at that state. `stress` = the **exact total derivative `dW/d(applied strain)`**, which by the envelope theorem at an equilibrium point reduces to a reaction-force-weighted sum over the Dirichlet-partition nodes (no re-solve needed). Verified against independent finite-difference re-solves to `1e-5`-`1e-7` relative error in `studies/fom_tangent_stability_test/`. | Historically produced by a generator script that no longer exists in this repository (see below); stored pre-computed in `pann/data/alltraj_stage10_direct_energy.npz` under the (confusingly plain) keys `train_strain`/`train_stress`/`train_energy` and `stage10_strain`/`stage10_stress`/`stage10_energy`. |

All of `pann/anisotropic`'s four PANN tiers are trained and evaluated against
the **third** row exclusively (the exactly energy-conjugate quantity) -- this
is a legitimate, self-consistent, and in fact more theoretically appropriate
choice for training an energy-based `S = dW/dE` model than the naive
volume-average, precisely because it is verified to satisfy the conjugacy
identity `dW = S . dE` to near machine precision along a real trajectory,
whereas the naive volume-average measurement does not (checked directly:
`30%` relative error against the same energy). This was confirmed
numerically during this project, not assumed.

## Why this matters for HPROM-ANN comparisons

`hprom/ann/`'s solver code computes the **measured/"legacy"** quantity
natively -- it has no concept of the "direct" energy-conjugate quantity at
all. Comparing its raw output directly against `pann/anisotropic`'s reported
Stage-10 numbers is therefore not apples-to-apples.

**This is now resolved for both HPROM-ANN operating modes.**
`pann/direct_energy/reaction_force_direct_stress.py` implements and verifies
the reaction-force procedure described below (relative L2 error against
known-good Stage-1 ground truth: `~1e-10`, effectively machine precision, on
three independently checked trajectories).
`pann/direct_energy/hprom_ann_direct_stress.py` applies it to a Stage-10
HPROM-ANN/D-HPROM-ANN run, reconstructing the full nodal displacement field
from each mode's saved reduced coordinates via the same LS decoder used
online (`u = Phi_m A_m q_m + Phi_s q_s(q_m)`, plus the macro-affine term on
free DOFs -- see that script's docstring for a debugging note on a real bug
caught in the first attempt at this reconstruction). Results for both modes
live in `pann/data/hprom_ann_direct_stage10_metrics.{json,npz}` and are
reported in `pann/anisotropic/PANN_anisotropic_claude.tex`'s master table.

`pann/data/stage10_hprom_direct_reference.json` is an older, pre-computed
D-HPROM-ANN reference (built by a since-lost script). The freshly computed
value **does not exactly reproduce it** (new/old ratio ~1.7x on aggregate
stress error, with a qualitatively different componentwise pattern -- shear
error improved ~9x, normal components ~2x worse). The most likely
explanation is the checkpoint-naming bug documented in the top-level
`README.md` (the plain `stage_7_ann_model_ls` vs. `_newton` mixup) affecting
whatever process produced the old reference, but this has **not** been
independently confirmed -- treat the old reference file as superseded, not
as a contradiction requiring further investigation before use.

## Reproducing the "direct" quantity for new data

Because `stress = dW/d(applied strain)` reduces to a reaction-force-weighted
sum at the Dirichlet partition (see table above), it can be computed cheaply
from **any** already-converged FOM solve -- no re-solving or finite
differences required:

1. Solve as usual (`fom_solver_rve.RunFomBatchSimulation`), which already
   computes the converged Gauss-point strain field and, via
   `PrecomputeDirichletPartitionFromNodes`, knows exactly which nodes are
   Dirichlet-constrained.
2. Extract `assembler._rhs` at those Dirichlet DOFs after calling
   `Assemble(u)` on the converged state -- this equals `-f_internal`
   everywhere (this problem has no external force term), which by
   equilibrium at a constrained node equals the reaction force there
   directly. No Kratos-native `REACTION` variable lookup is needed; the
   already-computed assembler state suffices.
3. Contract those reactions against the (affine, exactly known) sensitivity
   of the prescribed nodal displacement to the macro strain, `d(u_dirichlet)
   / d(applied strain)` (central finite difference on the cheap closed-form
   Dirichlet map -- not a re-solve).
4. Normalize by `thickness * A0` (`A0` = reference geometric area, no
   thickness factor -- see `reaction_force_direct_stress.py` for the exact
   sign/scale convention, which needed both a global minus sign and this
   specific normalization to match ground truth).

`pann/direct_energy/reaction_force_direct_stress.py` implements exactly
this, reusable for any future displacement history from this FOM.
`studies/fom_tangent_stability_test/run_fom_energy_hessian_audit.py`
independently cross-checks the same quantity via full finite-difference
re-solves (far more expensive, used only as an out-of-band verification
tool, not the production path).
