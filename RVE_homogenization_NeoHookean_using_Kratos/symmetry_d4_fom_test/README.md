# FOM test of the RVE square symmetry

This directory is an isolated experiment. It contains local copies of the RVE
mesh and Kratos input files; the existing ROM and PANN workflows are untouched.

The FOM input remains the existing Green--Lagrange strain vector

\[
\mathbf e = [E_{11},E_{22},\gamma_{12}], \qquad \gamma_{12}=2E_{12}.
\]

For a material symmetry operation \(\mathbf R\), the transformed input is

\[
\mathbf E' = \mathbf R^T\mathbf E\mathbf R.
\]

The FOM already reconstructs

\[
\mathbf C=\mathbf I+2\mathbf E,
\qquad
\mathbf F=\mathbf C^{1/2},
\]

so no change to the strain-based solver interface is needed.

The test checks the four distinct actions of the square group on a symmetric
second-order tensor: identity, reflection, a \(90^\circ\) rotation, and diagonal
reflection. The other four elements of \(D_4\) have the same action on
\(\mathbf E\) because \(\mathbf R\) and \(-\mathbf R\) give identical
\(\mathbf R^T\mathbf E\mathbf R\).

For every transformed state, the script compares the FOM result with

\[
\bar{\mathbf S}(\mathbf R^T\mathbf C\mathbf R)
=
\mathbf R^T\bar{\mathbf S}(\mathbf C)\mathbf R,
\qquad
\bar\Psi(\mathbf R^T\mathbf C\mathbf R)=\bar\Psi(\mathbf C).
\]

The energy is evaluated directly from the microscopic Neo-Hookean energy at the
converged Gauss points. It is not merely inferred from a plotted curve.

Run it with:

```bash
cd /home/kratos/ML_assisted_CLs_clean/RVE_homogenization_NeoHookean_using_Kratos/symmetry_d4_fom_test
source /home/kratos/set_up_kratos_eigen.sh
python3 run_d4_symmetry_test.py
```

All outputs are placed in `results/`. The key result is
`results/d4_symmetry_summary.md`.

The companion diagnostic can test the distinction from isotropy explicitly:

```bash
source /home/kratos/set_up_kratos_eigen.sh
python3 run_non_d4_rotation_diagnostic.py
```

It tests a \(45^\circ\) material rotation, which is deliberately not a
symmetry operation of a square RVE. Its results are written to
`isotropy_diagnostic/`.

## Informe explicativo

El informe didáctico se encuentra en `d4_symmetry_report.tex`; su PDF compilado
es `d4_symmetry_report.pdf`. Incluye las fórmulas, las figuras generadas desde
los resultados FOM y la consecuencia directa para la arquitectura de la PANN.

Tras repetir cualquiera de los experimentos, se actualiza con:

```bash
python3 make_report_figures.py
pdflatex -interaction=nonstopmode -halt-on-error d4_symmetry_report.tex
pdflatex -interaction=nonstopmode -halt-on-error d4_symmetry_report.tex
```
