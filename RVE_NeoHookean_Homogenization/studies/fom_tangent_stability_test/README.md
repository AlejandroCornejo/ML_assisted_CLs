# FOM tangent-stability audit

This is an isolated FOM experiment.  It answers one precise question before
we decide whether an SPD tangent should be imposed on a PANN:

\[
\text{Is } \mathbf D_E(\mathbf e)=\frac{\partial^2 W}{\partial \mathbf e^2}
=\frac{\partial \mathbf S}{\partial \mathbf e}
\text{ positive definite in the RVE states of interest?}
\]

The strain and stress convention is

\[
\mathbf e=[E_{11},E_{22},\gamma_{12}],\qquad \gamma_{12}=2E_{12},
\]

\[
\mathbf s=\frac{\partial W}{\partial\mathbf e}
= [S_{11},S_{22},S_{12}].
\]

This is deliberately **not** a derivative of Cauchy stress.  At finite strain,
the latter contains geometric transformations and is not the Hessian of the
stored energy.  The audit therefore measures the tangent relevant to our
energy-based PANN.

For each selected converged FOM state, the script uses a central 19-point
stencil.  At every point it re-solves the Kratos RVE by continuation from the
stored converged displacement, evaluates the microscopic Neo-Hookean energy
from the already converged FOM Gauss points, and forms the Hessian by finite
differences. Thus the reported matrix is symmetric by definition and has no
neural-network input.

Run:

```bash
cd /home/kratos/ML_assisted_CLs_clean/RVE_homogenization_NeoHookean_using_Kratos/fom_tangent_stability_test
source /home/kratos/set_up_kratos_eigen.sh
python3 run_fom_energy_hessian_audit.py --step 1e-2
python3 make_fom_tangent_figures.py --run h_1.000e-02
```

The default full audit launches one separate Python process per state, which
keeps Kratos' C++ memory use bounded and makes the audit resumable. Each
endpoint retains a log and a compact JSON record containing the energy.
`--keep-transient` is only useful when debugging old temporary histories.

To check the finite-difference step at an individual state, for example:

```bash
python3 run_fom_energy_hessian_audit.py --step 5e-3 --state extreme_biaxial_shear
```

The results are independent of any PANN/ICNN/ICKAN prediction.  They determine
whether globally imposing \(\mathbf D_E\succ0\) would agree with the FOM over
the sampled deformation envelope.
