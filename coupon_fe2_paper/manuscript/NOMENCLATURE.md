# PROM nomenclature and implementation map

Source convention: aresdeparga2026nonlinear. The mathematical definition,
not an older variable name in the code, controls the manuscript notation.
No solver variable or archived array was renamed as part of this revision.

| Manuscript symbol | Meaning | Current implementation |
|---|---|---|
| HDM / FOM | Unreduced microscopic problem | PeriodicRVE; FOM–FE2 remains the archived run label |
| N | Independent microscopic state dimension | 6320 after periodic identification and pinning |
| N_dof, d | Full nodal displacement dimension and physical field | 6480 components, including dependent/pinned entries |
| N_s, S_snap | Number and matrix of centered snapshots | 4950 affine-retaining independent displacement states |
| u_ref | Reference independent displacement state | Zero reference for this dataset |
| T, g_per(e) | Periodic identification and strain-dependent jump lift | PeriodicRVE.T and _g / _g_at |
| Phi, V_tra | Original POD ROB / conventional PROM ROB | Phi_ROM in decoder_basis_B_r39.npz; 39 columns |
| n_tra | Conventional PROM equilibrium dimension | 39 |
| V | Orthonormal primary ROB after strain-informed rotation | Phi_M; 3 columns |
| barred V | Orthonormal secondary complement in retained span | Phi_S; 36 columns |
| n, barred n | Primary and secondary dimensions | 3 and 36 |
| q | Orthonormal primary coefficients V-transpose (u - u_ref) | Reconstructed from A_M times deployed coordinates |
| barred q | Secondary generalized coordinates | q_S in the basis archive |
| T_m | Fitted map from original POD coefficients to strain | T_m |
| A_m | Invertible map from deployed coordinates to q | A_M |
| xi | Deployed primary coordinates, A_m-inverse q | q_M in the basis archive; often q in online routines |
| N(q) | Secondary-coordinate closure in source nomenclature | Nhat(A_m-inverse q) |
| Nhat(xi) | Closure in deployed coordinates | Network saved in nslave.npz |
| B(q) | Tangent of decoded independent displacement | value_and_jac, with the coordinate transform included |
| Z_r, Z_s | Residual and stress cubature supports | z_res and z_sig |
| w_e, v_e | Residual and stress multipliers | Fixed for HPROM, learned softmax fields for HPROM–ANN |
| D_h | Tangent of the complete effective stress map | Local implicit-function derivatives of actual residual/output |
| n_run | Number of timing repetitions | Kept distinct from the primary dimension n |

## Identities used in the manuscript

The source form is

    u_tilde(q) = u_ref + V q + Vbar N(q).

The actual deployment is

    u_tilde(xi) = u_ref + V A_m xi + Vbar Nhat(xi),
    q = A_m xi.

The iterative model solves for xi; the direct model prescribes xi = e.
For the training snapshots xi approximately equals e, but they are not
identical. The method-figure manifest quantifies the discrepancy.

The full physical field is

    d = T u + g_per(e).

The independent snapshots retain their affine part. A fluctuation
coordinate would be a = u - u_aff(e), with a correspondingly different
lift. Combining a basis from one convention with a lift from the other
would change the problem.

## Linearization distinctions

- B-transpose K B alone is the projected-stiffness term.
- The exact nonlinear Galerkin Jacobian also contains decoder curvature.
- Adaptive weights add weight derivatives to the residual Jacobian.
- The deployed microscopic iteration uses a modified-Newton matrix.
  The macroscopic tangent uses derivatives of the complete residual and
  stress output, through the implicit-function theorem. A consistent
  derivative is not a proof that the effective map has a scalar potential.

## Checked artifacts

The basis identities and coordinate discrepancies are checked by
build_method_figures.py. Its manifest records the basis hash. Relevant
implementation sources are build_decoder_basis.py, train_nslave.py,
numpy_decoder.py, maw_lab.py, maw_hprom_ann_law.py,
maw_hprom_ann_fast.py and direct_hprom_ann_law.py.
The paper describes these codes; this editorial revision does not modify them.
