# Anisotropic energy PANNs for the hyperelastic RVE

This directory contains the final non- (D_4) constitutive study.  It treats the
two-dimensional RVE as anisotropic, learns a scalar energy, and obtains the
energy-conjugate second Piola stress through

\[
\bm e=[E_{11},E_{22},\gamma_{12}]^T,qquad \gamma_{12}=2E_{12},qquad
\bm S=\frac{\partial W}{\partial\bm e}=[S_{11},S_{22},S_{12}]^T.
\]

The formal record is [PANN_anisotropic.pdf](PANN_anisotropic.pdf), generated
from `PANN_anisotropic.tex`.  It documents the assumptions, equations,
certificate, training protocol, FOM tangent audit, and Stage–10 results.

## Final models

| Model | Representation | Architectural guarantees |
|---|---|---|
| C-PANN | (a_0!cdot Ca_0, b_0!cdot Cb_0, a_0!cdot Cb_0, J) | Objectivity, (W(I)=0), (S(I)=0), energy–stress consistency. |
| PANN-T | (T\bm e, J), with (D_0=T^T\widehat D T) | Same guarantees; (T) is a local tangent preconditioner, not a stability proof. |
| PConv-PANN | balanced quartic and sixth-power norms of (Fd) and (operatorname{cof}(F)d), (J,J^2) | Objectivity, reference state, energy–stress consistency, anisotropy without a (D_4) average, polyconvexity, (J\to0^+) barrier, and growth at large (J). |

The PConv-PANN uses

\[
Q_{k,p}^{F}=\sum_i w_{ki}|Fd_{ki}|^{2p},\qquad
Q_{k,p}^{H}=\sum_i w_{ki}|\operatorname{cof}(F)d_{ki}|^{2p},
\qquad p\in\{2,3\},
\]

with (sum_iw_{ki}d_{ki}\otimes d_{ki}=I).  The sixth-power terms prevent
the accidental (90^\circ) symmetry that balanced quartic aggregates have in
two dimensions.  The proof is in the PDF and in `anisotropic_pann_model.py`.

## Reproducibility

All trainers read Stage–1 only.  The evaluator is the only script that opens
Stage–10.

```bash
python3 -B make_local_isotropic_mapping.py

python3 -B train_anisotropic_pann.py --kind free --epochs 500 --batch-size 4096 --threads 2 --seed 20260811
python3 -B train_anisotropic_pann.py --kind metric --epochs 500 --batch-size 2048 --threads 2 --seed 20260814
python3 -B train_anisotropic_pann.py --kind polyconvex --epochs 600 --batch-size 4096 --threads 2 \
  --polyconvex-widths 32,32 --seed 20260819

python3 -B evaluate_anisotropic_pann.py --kind all
python3 -B make_anisotropic_figures.py
python3 -B make_anisotropic_report.py
```

`checkpoints/` contains only the final three models.  `results/` holds their
training records, the Stage–10 predictions and metrics, and the LaTeX number
macros used by the memorandum.
