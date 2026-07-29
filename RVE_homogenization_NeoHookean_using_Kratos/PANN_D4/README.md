# PANN-D4: compact constitutive experiment

This is the only directory needed to read, run, and retrain the constitutive
PANN work. It contains three clear roles, not an uncontrolled sweep:

- `PANN-D4` is the selected high-accuracy energy surrogate used in the
  HPROM-ANN comparison.
- `PConv-PANN-D4` is one deliberately constrained extension that is
  polyconvex and volumetrically coercive by construction. It is retained to
  quantify the accuracy cost of those guarantees.
- `PConv-ICKAN-D4` is the improved polyconvex KAN extension. Its direct-input
  ICKAN companion is retained as a clearly labelled negative control.

## Selected PANN-D4

The input is the Green--Lagrange strain in physical component names:

\[
\mathbf e=[E_{xx},E_{yy},\gamma_{xy}]^T,
\qquad \gamma_{xy}=2E_{xy}.
\]

The scalar MLP is evaluated at the four distinct square-symmetry actions,

\[
\begin{aligned}
(E_{xx},E_{yy},\gamma_{xy}),\quad
(E_{yy},E_{xx},-\gamma_{xy}),\\
(E_{xx},E_{yy},-\gamma_{xy}),\quad
(E_{yy},E_{xx},\gamma_{xy}),
\end{aligned}
\]

and their energies are averaged. The reference correction makes
\(W(0)=0\) and \(\mathbf S(0)=0\). The predicted stress is always
\(\mathbf S=\partial W/\partial\mathbf e\); it is not a second neural
network.

Evaluate the selected PANN on the untouched Stage--10 path:

```bash
python3 -B evaluate_pann_d4.py
```

The expected errors are 0.052% in energy, 0.169% in global stress, and 0.785%
in shear stress. To repeat its fixed all-trajectory Stage--1 fit:

```bash
python3 -B train_pann_d4.py
```

This saves `checkpoints/PANN_D4_retrained.pt` and never overwrites the
distributed selected checkpoint.

## Polyconvex extension

The extension keeps the same external input \(\mathbf e\), but forms
\(C=I+2E\), \(J=\sqrt{\det C}\), and D4-closed directional measures

\[
q_a^F = |Fa|^2=a^TCa,
\qquad
q_a^H=|\operatorname{cof}F\,a|^2=a^T\operatorname{cof}(C)a.
\]

A positive-weight ICNN is convex and coordinate-wise increasing in these
features. Together with its D4 average and

\[
W_{\rm pc}=H_{D4}(E)-H_{D4}(0)-r\log J+\frac{\beta}{2}(J-1)^2,
\qquad r,\beta>0,
\]

this gives objectivity, D4 symmetry, reference conditions, energy--stress
consistency, polyconvexity, a barrier as \(J\to0^+\), and growth as
\(J\to\infty\). It does **not** imply convexity in \(E\), which is a
different condition.

Evaluate the trained polyconvex candidate and regenerate its audits:

```bash
python3 -B evaluate_polyconvex_d4.py
python3 -B make_polyconvex_d4_figures.py
```

This fixed candidate was trained only on Stage--1. It yields 10.751% energy
and 11.225% global-stress relative \(L^2\) error on the untouched Stage--10
path: it is a guaranteed constitutive baseline, not the selected high-fidelity
surrogate. To repeat its fit:

```bash
python3 -B train_polyconvex_d4.py
```

The run takes about 22 minutes on this CPU and writes
`checkpoints/PANN_D4_polyconvex_best.pt`.

## ICKAN re-evaluation

The historical ICKAN trajectory-3 reproduction test is not comparable with
the current energy-conjugate Stage--10 protocol. Both models here were instead
trained from scratch on all ten Stage--1 trajectories.

- `ICKAN-D4 direct` takes normalized `[E_xx,E_yy,gamma_xy]`, D4-averages the
  spline KAN, and applies the affine reference correction. It is a useful
  direct KAN control but is not polyconvex; its Stage--10 error is 23.008% in
  energy and 23.381% in global stress.
- `PConv-ICKAN-D4` takes positive affine scalings of four axial/diagonal
  `|F a|^2`, four `|cof(F)a|^2`, `J`, and `J^2`. With `base_fun=zero`, the
  ICKAN convex-monotone spline constraints, D4 average, `-r log(J)`, and a
  positive `(J-1)^2` term, it is polyconvex and volumetrically coercive. It
  gives 2.106% energy and 2.635% global-stress error on Stage--10.

Reproduce the fixed Stage--1 fits and then evaluate their saved checkpoints:

```bash
python3 -B train_ickan_d4.py --mode direct --width 8 --grid 15 --epochs 120 --batch-size 4096 --threads 1
python3 -B train_ickan_d4.py --mode minor_features --width 8 --grid 15 --epochs 100 --batch-size 4096 --threads 1
python3 -B evaluate_ickan_d4.py --checkpoint checkpoints/ICKAN_D4_direct_best.pt --output-prefix ICKAN_D4_direct_stage10
python3 -B evaluate_ickan_d4.py --checkpoint checkpoints/ICKAN_D4_minor_features_best.pt --output-prefix ICKAN_D4_minor_features_stage10
python3 -B make_ickan_d4_figures.py
```

Only the evaluator opens Stage--10. The adaptive multi-layer grid update of
the local ICKAN fork is intentionally disabled because it creates a NaN
reference derivative; the reported models use fixed, declared spline grids.

## Contents

- `PANN_D4.pdf` and `PANN_D4.tex`: formal technical memorandum and source.
- `pann_d4_model.py`, `train_pann_d4.py`, `evaluate_pann_d4.py`: selected
  high-accuracy PANN.
- `polyconvex_d4_model.py`, `train_polyconvex_d4.py`, and
  `evaluate_polyconvex_d4.py`: polyconvex candidate and its audit.
- `make_polyconvex_d4_figures.py`: figures cited in the memorandum.
- `ickan_d4_model.py`, `train_ickan_d4.py`, and `evaluate_ickan_d4.py`:
  direct ICKAN control and the PConv-ICKAN candidate.
- `make_ickan_d4_figures.py`: ICKAN comparison and barrier figures.
- `data/`: compact direct FOM labels for Stage--1 and Stage--10.
- `checkpoints/`: selected PANN, PConv-ICNN and the two declared ICKAN
  checkpoints.
- `results/`: reproducible metrics, history, predictions, and comparisons.

The stored legacy HPROM-ANN stress arrays use a historical volume-average
measure. They are not silently ranked by the evaluators against the PANN target
\(\mathbf S=\partial W/\partial\mathbf e\). The compatible direct HPROM-ANN
reference used in the PDF is
`results/stage10_hprom_direct_reference.json`.
