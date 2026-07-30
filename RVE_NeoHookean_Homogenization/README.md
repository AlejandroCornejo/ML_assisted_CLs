# RVE NeoHookean Homogenization

A 2D nonlinear (compressible Neo-Hookean) hyperelastic representative-volume-
element (RVE) homogenization pipeline: full-order model (FOM), a projection-
based reduced-order model (PROM) with an ANN-based nonlinear manifold
(PROM-ANN), its hyper-reduced counterpart (HPROM-ANN, via empirical cubature
/ MAW-ECM), and a family of physics-augmented neural-network (PANN)
constitutive surrogates trained on the same RVE.

This is a cleaned, reorganized successor to the original flat research
repository. Only the ANN-based surrogate family is included here (GPR, RBF,
and POD-DL manifold variants, and their hyper-reduced counterparts, remain in
the original repository, along with unrelated collaborator experiments and
tangential paper/thesis material).

## Layout

```
core/           Shared FOM solver, HPROM base class, ECM/plot utilities, mesh + material files.
mawecm/         MAW-ECM weight-evaluation utilities used by hprom/ann.
trajectories/   Stage-1 training trajectory generation + saved FOM snapshots (3.5G).
pod/            POD basis construction (free-fluctuation modes) + PROM verification.
prom/
  pod/          Plain POD-Galerkin PROM (no manifold surrogate) -- prom_solver_rve.py.
  ann/          PROM-ANN: ANN-based nonlinear manifold (LS secondary-coordinate closure), training + benchmark.
hprom/
  ann/          HPROM-ANN: hyper-reduced PROM-ANN via ECM/MAW-ECM. Two canonical ECM configurations
                (see below), both real and both still meaningful, not one superseding the other.
pann/
  anisotropic/  The current, actively-developed PANN work: a 4-tier ablation
                (direct regression -> free hyperelastic -> polyconvex ICNN -> polyconvex ICKAN)
                for a general anisotropic RVE, with no material-symmetry assumption. See
                pann/anisotropic/PANN_anisotropic_claude.tex for the full technical memorandum.
  data/         Shared Stage-1/Stage-10 training and evaluation data consumed by pann/anisotropic
                (see DATA_DICTIONARY.md -- this is the file whose two different stress
                definitions caused a real investigation during this project).
benchmarks/     FOM-solver cross-checks (vectorized vs. Kratos-native) and profiling, independent of any surrogate family.
studies/        fom_tangent_stability_test: is the FOM's own material tangent SPD? -- no, and that's
                expected; directly cited by pann/anisotropic's memo. (symmetry_d4_fom_test, which checked
                whether this RVE's FOM response happens to be D4-symmetric, was built to justify the old
                PANN_D4 approach's D4-symmetric model and is not read by anything here -- it remains only
                in the original repository.)
animations/     WCCM2026 conference presentation scripts and final rendered assets. See animations/README.md
                for its own, narrower migration scope.
```

## Canonical vs. superseded (read before reusing a checkpoint or ECM directory)

- **PROM-ANN checkpoint**: `prom/ann/stage_7_ann_model_ls` (validation error
  2.61e-4) is canonical. During migration this was renamed from
  `stage_7_ann_model_ls_newton`; the previous plain `stage_7_ann_model_ls`
  (validation error 3.28e-4, an earlier and less accurate training run) was
  **not** migrated. Some WCCM2026 animation scripts previously pointed at the
  wrong (superseded) one under the old name -- this was a real, previously
  undetected inconsistency, fixed as part of this migration (see
  `animations/README.md`).
- **HPROM-ANN ECM configuration**: two configurations are both genuinely
  meaningful, not a "current vs. old" pair:
  - `hprom/ann/ecm_fixed/` -- classical fixed empirical-cubature weights
    (`--hprom-homogenization-mode ecm_fixed`, the script's own default).
  - `hprom/ann/maw_dynamic/` -- adaptive MAW-ECM homogenization weights
    (`--hprom-homogenization-mode maw_dynamic`); this is also a *build input*
    dependency of `ecm_fixed`'s raw dataset, not a downstream replacement of it.
- Not migrated (confirmed dead, zero references anywhere in the code):
  `stage_9_hprom_ann_data_ls_4hom_split_hom50_sum990` (an abandoned weight-set
  experiment). Its similarly-named `stage_9_ecm_dataset_ann_ls_hom50` raw
  dataset is a *different* thing and **is** migrated (`hprom/ann/
  stage_9_ecm_dataset_ann_ls_hom50/`, 3.0G) -- checking the provenance
  metadata baked into `maw_dynamic/ecm_weights_all.npz` (`data_dir` field)
  showed it is the actual build input for the canonical `maw_dynamic`
  configuration, not a byproduct of the dead experiment; an earlier pass of
  this migration wrongly excluded it based on the naming coincidence alone.
  `ecm_fixed/` was built from the (correctly migrated) plain
  `stage_9_ecm_dataset_ann_ls/` instead.

## What's deliberately not here

Full accounting is in the migration plan, but briefly: `bakckup_21_modes/`
and `backup_nonvectorized_proms_and_fom/` (explicit backups of superseded
pipeline generations), GPR/RBF/DL surrogate families and their hyper-reduced
counterparts, dead code with unrunnable hardcoded paths
(`ICKAN_surrogate.py`, the `*_rbf_ls` scripts that reference data no longer
on disk), an orphaned generic KAN checkpoint (`model/`), a collaborator's
separate ICKAN benchmarking tree (`Sebastian_ICKAN_Tests/`), and unrelated
paper/thesis content bundled under the old `Paper_Manuscript/`. None of these
were deleted -- they remain in the original
`RVE_homogenization_NeoHookean_using_Kratos/` repository.

Within what *was* migrated, exploratory/intermediate checkpoints and result
files that are not read by any current script or macro-generation pipeline
were also left behind (e.g. `pann/anisotropic/checkpoints/` keeps only the
six checkpoints actually referenced by `make_polyconvex_report_claude.py`,
not the dozen-plus hyperparameter-search runs that produced them).

## Resolved during this migration

`pann/anisotropic`'s Stage-10 comparison against both HPROM-ANN operating
modes is now complete (`pann/direct_energy/`, see `DATA_DICTIONARY.md`) --
the master results table in `PANN_anisotropic_claude.tex` no longer has a
pending row. Getting there surfaced two further, real inconsistencies (not
migration artifacts, genuine pre-existing gaps this reorganization happened
to uncover), both documented in `DATA_DICTIONARY.md`: the freshly computed
D-HPROM-ANN number does not exactly reproduce an older, now-superseded
reference value, and the raw ECM dataset actually needed to rebuild the
`maw_dynamic` HPROM-ANN configuration was initially (incorrectly) excluded
from this migration based on a naming coincidence before being restored.
