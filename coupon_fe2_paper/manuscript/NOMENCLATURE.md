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
| n_tot | Total retained span dimension, n + barred n | 39, equal to n_tra in this comparison |
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
| B_xi | Decoder tangent in deployed coordinates | B(q) A_m |
| N_el | Number of full-mesh elements; multiplier normalization | n_full_elements / target_sum, not physical volume |
| Z_r, Z_s | Residual and stress cubature supports | z_res and z_sig |
| w_e, v_e | Residual and stress multipliers | Fixed for HPROM, learned softmax fields for HPROM–ANN |
| D_h | Tangent of the complete effective stress map | Local implicit-function derivatives of actual residual/output |
| s_h | Three-component reduced effective stress output | Diagonal entries of Fbar-inverse Pbar, plus the average of its two shear entries |
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
- For fixed-weight affine HPROM, stiffness and analytic lifting derivatives
  yield the state sensitivities; the stress derivative uses a central
  difference along those sensitivities. For HPROM–ANN, the partial
  derivatives of the complete residual/output are central-differenced.
  The direct mode central-differences its full strain-to-stress map.

The input-informed identification and the option of direct evaluation are
introduced alongside MAW–ECM in its current manuscript; they are decoder
choices, not intrinsic parts of the cubature algorithm. Our primary/secondary
symbols remain those of aresdeparga2026nonlinear.

Correction to the initial Section 3 audit: the deployed supports do undergo
adaptive-weight pruning, rather than fixed-ECM selection directly at their
final size. The residual support in maw_res_long10.npz equals that in the
archived maw_phase2_res.npz. However, the available sweep_phase2.py does not
pass K_graph or enable use_global_graph_2ndstage; in the available pruning
library its smooth_laplacian_all_iterations option forces the local active-set
phase, not graph coupling. The discarded graph-construction return and script
comments do not establish historical graph regularization. Historical pruning
diagnostics are absent from these checkpoints; this provenance gap remains
explicit in REFERENCE_AUDIT.md.

## Checked artifacts

The basis identities and coordinate discrepancies are checked by
build_method_figures.py. Its manifest records the basis hash. Relevant
implementation sources are build_decoder_basis.py, train_nslave.py,
numpy_decoder.py, maw_lab.py, maw_hprom_ann_law.py,
maw_hprom_ann_fast.py and direct_hprom_ann_law.py.
The paper describes these codes; this editorial revision does not modify them.

# Manuscript glossary (style pass, agreed 2026-09-27)

Every fragment of the style pass applies these names. The last column lists the
variants to search for before a fragment is closed.

| Object | Canonical form | Replace |
|---|---|---|
| Energy without curvature constraints | Unconstrained energy (capitalized, abstract included) | flexible energy reference, flexible energy-based reference, unconstrained energy |
| ICNN and ICKAN energies together | constrained energies ("polyconvex" only as an adjective) | polyconvex energies/models as a name |
| ICNN, ICKAN and Unconstrained together | learned energies | PANN evaluations/times |
| Periodic microscopic problem | periodic (e.g. periodic microscopic reference) | constrained discretization/tangent/residual |
| Cells | MC-RVE, SC-RVE (defined in the Introduction roadmap) | multicavity cell, single-cavity cell |
| Region of the data | fitting box; "beyond the data" = outside the fitting box | box (undefined), training box, sampling box, engineering-strain box, strain domain |
| Probe distances | overshoot factors | ring factors, probe rings, ringwise |
| Two-scale scheme | FE² (any micro evaluator); nested FOM–FE² is the reference | "is not itself" FE² |
| Unreduced microscopic model | FOM, always | HDM |
| Linear-basis reduced model | affine HPROM at first mention per section, then HPROM | linear HPROM, conventional HPROM, D--HPROM--ANN |
| $\varepsilon_{\rm SVD}$ | integrand-SVD truncation: smallest rank whose relative truncation norm (as $\epsilon_{\rm POD}$, Eq. 46) does not exceed it (Sec. 4.4.1, App. E; code: rank_for) | undefined tolerance |
| Reduced models, collectively | PROM(s) for the method class (defined in the abstract and Sec. 1.3); the HPROMs for the three deployed models (affine HPROM, HPROM--ANN, D-HPROM--ANN) | reduced microscopic models/solvers/references, microscopic surrogates, reduced evaluators, intrusive reduced references, reduced model(s) |
| Acronyms | only acronyms reused later: ECSW, ECM, MAW--ECM (en dash), HPROM; ICNN/ICKAN/POD expanded once in the body | DEIM/EQP/CECM/SAW acronyms (never reused), repeated expansions |
| Feature scale | tau_i | s_i as a scale |
| Fixed-rule cubature states | N_c = 495 (every tenth training state) | undefined N_c |
| Stress names | first/second Piola–Kirchhoff stress, full name always (as in As'ad, Klein, Linden) | "first/second Piola stress" shorthand |
| Section 5.2 headings | Number of paired features; Effect of learning the features; Admissibility beyond the data | question headings, "answers this question" |
| Figure references | Figure~ at sentence start, Fig.~ elsewhere | mid-sentence Figure~ |
| Citations | Author et al.~\cite{} kept; no Ref./Refs. | Ref.~, Refs.~ |
| Author name | Ares de Parga, everywhere including references | Ares De Parga |
| Tangent $\bm D=\partial\ssv/\partial\ee$ | constitutive tangent (Sec. 2, Eq. 3) | material tangent, strain tangent |
| Untrained feature controls | fixed features, fixed paired directions (pairs with learned) | prescribed features/directions |
| Feature scale | $\tau_i$ in $y_i=(z_i-z_i^{\rm cen})/\tau_i$ | $s_i$ (clashes with stress components) |
| $H(\bm z)=\widehat H(\bm y(\bm z))$ | learned contribution ($\widehat H$: neural core) | neural contribution, core contribution, scalar core $H$ |
| Spelling | labeled; 100~kN | labelled; 100 kN |
| Cell side $\ell$ | defined at first use in the Fig. 5 caption; normalizes $\|\bm w\|$ (Fig. 5) and $X_i$ (Figs. 6, 10). Also the layer index of Sec. 3.4, kept because the contexts are separate | undefined $\ell$ |
| $\bm w$ | fluctuation, Eq. (5) (Sec. 2 and Fig. 5, whose caption cites the equation). Also the ICNN/ICKAN skip weights (Sec. 3.4) and the cubature weights (Sec. 4.4, App. E), each defined where it is used | |
