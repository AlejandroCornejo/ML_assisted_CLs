# Stage 00 results

Cell: square side 2, one rotated elliptical hole. Porosity 20% (19.968%
realised), a/b = 2, rotated 30 deg. Matrix E = 1.628 GPa, nu = 0.4, plane
strain, T6 elements. Homogenization denominator = CELL area (void included).

## Mesh convergence of C0 (affine BCs, 1x1)

| size_far | size_hole | elements | nu_xy | g12/E11 | dC0 vs coarser |
|---|---|---|---|---|---|
| 0.300 | 0.140 | 375 | 0.4991 | -0.1863 | -- |
| 0.220 | 0.100 | 673 | 0.4992 | -0.1865 | 4.75e-04 |
| 0.140 | 0.050 | 2466 | 0.4993 | -0.1866 | 3.60e-04 |
| 0.100 | 0.035 | 4987 | 0.4993 | -0.1866 | 2.66e-05 |

C0 is converged at the coarsest mesh: 13x more elements move it 0.08% in
total. Note this is a ZERO-STRAIN quantity and converges faster than the
finite-strain response, where strain concentrates at the pore rim -- so this
is necessary but not sufficient, and the mesh must be re-checked at a
representative finite strain before deployment. STILL OPEN.

## Affine-Dirichlet boundary layer (cell-count study, coarse mesh)

| n | elements | Ex (GPa) | nu_xy | g12/E11 |
|---|---|---|---|---|
| 1 | 375 | 1.26502 | 0.4991 | -0.1863 |
| 2 | 1260 | 1.23980 | 0.4862 | -0.2419 |
| 3 | 2631 | 1.23054 | 0.4813 | -0.2644 |
| 4 | 4546 | 1.22582 | 0.4788 | -0.2760 |
| 5 | 6897 | 1.22299 | 0.4774 | -0.2830 |

Ex decreases monotonically, the direction the ordering C_Neumann <=
C_periodic <= C_affine requires. A least-squares fit of `value = limit + B/n`
over n = 2..5 is essentially exact:

| quantity | limit | B | R^2 |
|---|---|---|---|
| g12/E11 | -0.31024 | +0.13688 | 0.999923 |
| Ex (GPa) | +1.21182 | +0.05601 | 0.999975 |
| nu_xy | +0.47149 | +0.02942 | 0.999941 |

The coarse mesh reproduces the fine mesh on a BLOCK too: at n = 3, coarse
(2631 elem) gives Ex 1.23054, g -0.2644 against fine (16864 elem) 1.22983,
-0.2651, i.e. 0.06% and 0.3%. Per-cell density carries over.

**The consequence that decided the formulation.** Coupling error = 0.441/n
while element count grows as n^2, so cost ~ 1/error^2:

| target error in g12/E11 | n | elements | est. FOM-FE^2 reference |
|---|---|---|---|
| 8.8% | 5 | 6,897 | ~10 h |
| 5% | 9 | ~22,700 | ~33 h |
| 2% | 22 | ~135,000 | ~8 days |
| 1% | 44 | ~542,000 | ~33 days |

Affine BCs cannot be converged by brute force. Periodic BCs give the
converged answer on ONE cell, which is why they are the standard here.

## Periodic C0, and the two verification gates

Independent linear-elastic T6 assembly (`periodic_C0.py`), not Kratos, so that
the affine mode provides a cross-check against the validated path.

**Gate 1 -- this assembly with AFFINE BCs vs Kratos's analytic tangent at
E = 0.** Validates the assembly, the Voigt convention, and the claim that a
hyperelastic tangent at zero strain equals the linear elastic stiffness.

| | this code | Kratos | rel |
|---|---|---|---|
| Ex | +1.265149e9 | +1.265020e9 | 1.02e-04 |
| nu_xy | +0.4990971 | +0.4991 | 5.75e-06 |
| g12/E11 | -0.1861788 | -0.1863 | 6.51e-04 |

**Gate 2 -- PERIODIC BCs on one cell vs the affine 1/n extrapolation.** Two
fully independent routes to the same numbers.

| | periodic, 1 cell | 1/n extrapolation | rel |
|---|---|---|---|
| Ex | +1.211576e9 | +1.211820e9 | 2.02e-04 |
| nu_xy | +0.4717525 | +0.4714900 | 5.57e-04 |
| g12/E11 | -0.3108176 | -0.3102400 | 1.86e-03 |

Agreement to 0.02% in Ex and 0.2% in the coupling. This confirms the 1/n law
is the correct asymptotic rather than a lucky fit, confirms the periodic
implementation, and settles the converged effective properties.

Cost: **253 elements, 565 nodes** for the exact answer, against ~135,000 for
2% by brute force, and against 990 in the previous project's RVE.

## The converged effective material

```
C0 (Pa) = [[1.5627e9  6.2693e8  1.7815e8]
           [6.2693e8  1.2176e9  1.6897e8]
           [1.7815e8  1.6897e8  3.1671e8]]
```

* symmetric to 9.3e-16; eigenvalues 2.813e8, 7.414e8, 2.074e9 (positive definite)
* **|C13|/C11 = 11.4%, |C23|/C22 = 13.9%** -- the group-free signature, and
  STRONGER than affine 1x1 reported (8.5%, 9.9%). Affine BCs on a single cell
  suppress the coupling by 40%.
* Ex = 1.2116 GPa, Ey from C0 as needed, nu_xy = 0.4718
* **g12/E11 = -0.311**: an axial macro load produces shear at 31% of the axial
  strain. This is what makes the smooth ASTM coupon a non-trivial FE^2 problem
  -- the richness comes from the material, not from a stress concentration.

Transverse budget implied by nu_xy = 0.4718 (linear limit):

| E11 | E22 |
|---|---|
| 0.10 | -0.047 |
| 0.20 | -0.094 |
| 0.30 | -0.142 |

## Stage 00B: nonlinear periodic solve

`periodic_fom.py`. Kratos's validated `VectorizedAssembler` does all the
physics (true Neo-Hookean, total Lagrangian, R = -f_int); only the constraint
handling is new, as an elimination `u = T u_ind + g(E)` with Newton run in the
reduced space `(T^T K T) du_ind = T^T R`.

This is the SAME decomposition the reduced-order machinery already uses,
`u = u_affine(E) + Phi q`, which is the textbook periodic split. KUBC is the
special case where T is a plain selection onto interior dofs. Nothing about
the ROM ansatz changes, and a POD basis built from periodic snapshots spans
periodic fluctuations automatically, periodicity being a linear constraint.

Mesh: 565 nodes, 253 elements -> 1130 dofs, 1058 independent, 70 tied, 2 pinned.

**Gate 3 -- nonlinear periodic tangent at small strain vs the linear periodic
C0** (itself verified two independent ways by gates 1 and 2).

| | |
|---|---|
| max relative difference | **1.835e-04** |
| cost of the full 3x3 tangent | 0.1 s for 7 solves |
| cold solve at E11 = 0.20 | 1.2 s |

The three gates form a chain from the new periodic solver back to the
project's own validated Kratos path: 1.0e-04, 2.0e-04, 1.8e-04.

## The true uniaxial-tension path

Solved rather than imposed: `S22(E) = 0` and `S12(E) = 0` for (E22, gamma12)
at each E11, by a 2x2 Newton on the macro tangent. With shear-normal coupling
present, true uniaxial STRESS requires gamma12 != 0 -- the material shears
under a purely axial load, which is the design premise.

| E11 | E22 | gamma12 | S11 (Pa) | secant nu |
|---|---|---|---|---|
| 0.05 | -0.02292 | -0.01567 | 5.522e7 | 0.4584 |
| 0.10 | -0.04455 | -0.03159 | 1.012e8 | 0.4455 |
| 0.15 | -0.06494 | -0.04773 | 1.396e8 | 0.4329 |
| 0.20 | -0.08418 | -0.06401 | 1.720e8 | 0.4209 |
| 0.25 | -0.10233 | -0.08034 | 1.995e8 | 0.4093 |
| 0.30 | -0.11948 | -0.09663 | 2.229e8 | 0.3983 |
| 0.40 | -0.15110 | -0.12874 | 2.605e8 | 0.3778 |
| 0.50 | -0.17962 | -0.15974 | 2.889e8 | 0.3592 |
| 0.60 | -0.20555 | -0.18922 | 3.108e8 | 0.3426 |

Three findings:

1. **The secant transverse ratio DECREASES with strain**, 0.458 -> 0.343. So
   the linear-limit nu_xy = 0.4718 is the worst case and the material becomes
   progressively less contractive as it stretches. At E11 = 0.30 the real E22
   is -0.119, against -0.142 from the linear estimate. The transverse budget
   is more comfortable than the linear figure implied.
2. **gamma12/E11 stays at -0.313 ... -0.315 up to E11 ~ 0.6**, against -0.3108
   predicted by C0, then drifts down: -0.291 at E11 = 1.0, -0.249 at 1.7. So
   the anisotropy signature persists well past the small-strain regime but is
   not constant over the whole range.
3. **No failure through E11 = 1.70** (E22 = -0.398, gamma12 = -0.423). The
   usable tensile range was not reached.

Continued to E11 = 1.70:

| E11 | E22 | gamma12 | S11 (Pa) | secant nu |
|---|---|---|---|---|
| 0.80 | -0.25124 | -0.24312 | 3.420e8 | 0.3141 |
| 1.00 | -0.29068 | -0.29061 | 3.627e8 | 0.2907 |
| 1.20 | -0.32547 | -0.33284 | 3.772e8 | 0.2712 |
| 1.50 | -0.37108 | -0.38906 | 3.921e8 | 0.2474 |
| 1.70 | -0.39801 | -0.42325 | 3.991e8 | 0.2341 |

### Why the pore never closes, and what that changes

**Pore closure is a compression phenomenon, not a tension one.** Under
stretching the pore ELONGATES; ligaments thin but never come into contact.
And because the macro deformation is close to affine, the pore contracts in
proportion to the cell, so the ligament ratio is preserved.

That reinterprets the previous project's -0.1 limit: it was never a bound on
E22 alone, it was a bound on a COMPRESSIVE state. Here E22 reaches -0.398
without difficulty because it is accompanied by E11 = +1.70, so the cell is
globally stretched and the pore is open.

**Consequence for the design.** The binding constraint is no longer what the
RVE can survive. It is the validity of first-order homogenization and the
physical reasonableness of the strain level. The guard recorded in the design
document -- the load level must not creep -- becomes the real constraint, and
it is a principled one rather than an artefact of cell geometry. This is the
healthy situation, and it is the opposite of what drove the previous attempt.

## Mesh convergence at FINITE strain, and the deployed mesh

The open question from the E = 0 study: C0 is a zero-strain quantity and
converges faster than the nonlinear response, where deformation localises at
the pore rim. Re-checked ON the true uniaxial path, at E11 = 0.10, 0.20, 0.30,
relative to a 1546-element reference:

| elements | dE22 | dgamma12 | dS11 | worst |
|---|---|---|---|---|
| 134 | 3-7e-04 | 5.1-5.7e-03 | 0.9-1.2e-03 | **0.57%** |
| 253 | 4.7-6.9e-04 | 1.0-1.6e-03 | 3.0-4.2e-04 | **0.16%** |
| 467 | 1.2-1.9e-04 | 3.6-5.4e-04 | 1.0-1.4e-04 | **0.054%** |

gamma12 is the most sensitive quantity, as expected -- it is the smallest and
the most geometry-sensitive. Note dE22 is NOT monotone (134 beats 253 at
E11 = 0.10, both at the 1e-4 level). That is not an error: it means the
plateau has been reached where mesh topology dominates over resolution, which
is itself evidence of convergence.

## Local field convergence, and the mesh actually deployed

The homogenized-average criterion above is NOT the right one for choosing the
mesh, and measuring showed why. C0 and the homogenized stress are volume
AVERAGES, and averages converge much faster than the fields they average.
POD and the ECM operate on the FIELD.

Measured: `||P||` (first Piola-Kirchhoff, `P = F S`) at every Gauss point on
the true uniaxial path at E11 = 0.20, relative to a 3095-element reference.
P is chosen deliberately -- it is the integrand of the homogenized output in
first-order homogenization, and the same output of interest Hernandez uses in
the MAW-ECM metamaterial benchmark, so the field being converged is the one
the cubature rule must integrate. All statistics are mesh-independent
functionals, so different topologies are comparable.

| elements | GPs | **max** | p99 | p90 | L2 | **S11 (average)** |
|---|---|---|---|---|---|---|
| 134 | 402 | 2.62e-02 | 2.82e-02 | 8.2e-03 | 1.16e-03 | 9.8e-04 |
| 253 | 759 | **1.82e-02** | 4.94e-03 | 4.9e-03 | 4.03e-04 | **3.5e-04** |
| 467 | 1401 | 1.49e-02 | 4.43e-03 | 3.5e-03 | 1.43e-04 | 1.2e-04 |
| **1546** | **4638** | **4.86e-03** | 3.68e-04 | 1.2e-04 | 1.26e-05 | 9.9e-06 |

At 253 elements the average is converged **52x better than the local
maximum** (3.5e-04 against 1.82e-02). And 467 is a bad deal: the maximum only
falls 1.82% -> 1.49% for 1.85x the points, so the real choice was 253 or 1546.

**Deployed: 1546 elements** (`MESH_SIZE_FAR = 0.14`, `MESH_SIZE_HOLE = 0.05`,
both in `config.py` rather than duplicated across callers). 3240 nodes, 6480
dofs, **6320 independent**, 160 tied. Local field maximum at 0.49%, average
at 0.001%. That 4638 integration points lands at 0.88x the 5260 of
Hernandez's benchmark is a welcome consequence, not the reason.

### Gates re-run on the deployed mesh

| gate | quantity | rel |
|---|---|---|
| 1 | Ex / nu_xy / shear vs Kratos affine | 4.0e-04 / 3.4e-04 / 1.8e-03 |
| 2 | Ex / nu_xy / shear vs 1/n extrapolation | 8.4e-04 / 7.7e-04 / 4.5e-03 |
| 3 | nonlinear periodic tangent vs linear periodic C0 | **6.3e-05** |

Read these correctly. Gates 1 and 2 compare a FINE-mesh computation against
references computed on the COARSE mesh (the Kratos affine reference came from
the 375-element mesh, and the 1/n extrapolation from the coarse-density
cell-count study), so their residual reflects mesh refinement rather than
implementation error, and it sits at the known mesh-sensitivity level -- the
shear coupling being the most mesh-sensitive quantity, as the table above
shows. Gate 3 is the one comparing like with like on the same mesh, and it
IMPROVED from 1.8e-04 to **6.3e-05**.

### Deployed effective properties (periodic, 1546 elements)

| | |
|---|---|
| Ex | 1.210799e9 Pa |
| nu_xy | 0.4718511 |
| gamma12/E11 | **-0.3116439** |

## Stage 00: closed

| item | outcome |
|---|---|
| Cell geometry, rotated ellipse | 20% porosity, a/b = 2, 30 deg |
| Mesh | 253 elements, converged at E = 0 and at finite strain |
| BCs | periodic, three gates at ~2e-4 against the validated path |
| Homogenization denominator | thickness x CELL area (void included) |
| C0 | measured, symmetric, positive definite |
| Group-free check | PASSED, |C13|/C11 = 11.4%, |C23|/C22 = 13.9% |
| Anisotropy | gamma12/E11 = -0.31, persists to large strain |
| Matrix locking check | nu = 0.4 gives K/G = 4.67, no risk with T6 |
| Usable tensile range | >= 1.70 in E11, limit not reached |
| HPROM compatibility | ROM ansatz unchanged; ECM/MAW-ECM BC-independent |

Cost per RVE query, measured: cold solve at E11 = 0.20 in 1.2 s; the full 3x3
macro tangent by finite differences in 0.1 s for 7 warm-started solves.

## Bugs found and fixed

1. **OCC bounding boxes.** OCC inflates them by ~1e-7, so the original 1e-9
   box test classified NO curve as an outer edge, and the refinement field was
   refining the outer boundary along with the pore rim. Replaced by a
   centre-of-mass test, which is unambiguous. This wasted elements but did not
   invalidate the convergence study, which was verified on its own terms.

2. **Homogenization denominator missing the thickness.** Gate 3 first failed
   by a factor that was uniformly 20.0 across all nine components of the
   tangent -- a scale factor, not physics. `w_detJ` already carries the
   element thickness (`sum(w_detJ)` came out 0.16 = 0.05 * 3.2, thickness
   times solid area), so the denominator is `thickness * A0`, exactly the
   product the project's own laws use as `thickness * hom_reference_measure`.
   1/0.05 = 20. Worth recording because the failure signature -- a single
   uniform ratio across every component -- diagnosed it immediately, where a
   scattered mismatch would have meant a real formulation error.

## Still open in this stage

1. Mesh convergence at a representative FINITE strain, not just at E = 0.
2. Usable tensile range: where pore closure or ligament lock-up begins.
3. Stage B: nonlinear periodic solve plus a macro tangent. The finite-
   difference tangent route already exists and is validated, and needs no new
   derivation; the analytic tangent under constraints stays an optimisation,
   not a prerequisite.
