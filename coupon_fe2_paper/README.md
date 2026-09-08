# Problem 1: ASTM D638 tensile coupon on a rotated-elliptical-hole RVE

Design document. Written BEFORE any data is generated, deliberately: the
failure of the previous attempt was ordering, not execution. There, the
sampling box was chosen from what the RVE could survive, and a macro problem
was then hunted that fit inside it. Every dimension of that macro problem was
therefore a fitted parameter, and none of it was defensible on its own terms.

Here the causality runs the other way. The macro problem is fixed first, from
a standard. The strain envelope it visits is then *measured*. Only then is the
training domain defined, as that envelope plus margin.

## The message

Faisal As'ad's message, and ours: **accuracy does not certify admissibility.**
A surrogate can match stress data closely and still not be a material.

There is a trap in this that shapes the whole experiment. Once training is
designed for the problem, *every* tier becomes accurate inside the envelope,
including the uncertified ones. So if the paper only reported in-envelope
accuracy, it would report that everything works, which is not a result.

As'ad's failures do not come from in-domain accuracy either. They come from
two other places, and he separates them explicitly:

1. **Outside the training subdomain** (his Figs. 6-8 distinguish inside from
   outside by construction).
2. **Stability and consistency**, not accuracy: a membrane given only an
   initial velocity that must not deform on its own; Newton convergence on a
   pressurised cylinder; bounded kinetic energy in free oscillation.

So Problem 1 is a pair, and both halves are required:

- **1a, in-envelope validation.** Training designed for the problem; all tiers
  compared against nested-FOM truth. This is the robustness half, and it is
  what makes the comparison fair and reproducible.
- **1b, the probe.** Load pushed beyond the trained envelope, *and* a
  stability/consistency test in As'ad's sense. This is where the certificate
  earns its place: the certified tiers hold, the uncertified ones do not.

Scope is stated, not implied: this demonstrates surrogate robustness under
first-order homogenization in a tension-dominated regime. It is not a claim
that first-order homogenization is valid at extreme deformation.

## Macro geometry: ASTM D638 Type I

Taken from the standard, read directly from D638-14 rather than from
secondary sources:

| | |
|---|---|
| W, width of narrow section | 13 mm |
| L, length of narrow section | 57 mm |
| WO, width overall | 19 mm |
| LO, length overall | 165 mm |
| G, gage length | 50 mm |
| D, distance between grips | 115 mm |
| R, radius of fillet | 76 mm |
| T, thickness | 3.2 mm |

Derived: **D/d = 19/13 = 1.462**, **r/d = 76/13 = 5.846**.

Why this geometry rather than a plate with a central hole, or the hand-tuned
dogbone that preceded it:

- **The standard is designed so the shoulder does not concentrate stress.**
  A coupon must fail in the gage section, where the extensometer sits, not at
  the transition. That is exactly the property we need, and it comes from the
  standard's purpose rather than from a parameter we chose. The previous
  dogbone had r/d = 0.5 and D/d = 1.0625: a tight fillet on a nearly invisible
  taper, both fitted.
- **The gradient is gentle**, which is what first-order homogenization needs.
  A central circular hole has a better-characterised concentration factor
  (Kirsch/Howland, one parameter) but its steepest gradient sits exactly at
  the hole edge, i.e. precisely where the peak strain of interest is. That is
  a validity cost paid for a convenience gain, and validity wins.
- Our own earlier shoulder sweep supports the choice: at fixed D/d = 1.5,
  raising r/d through 0.5 -> 1.0 -> 2.0 drove the shear ratio g12/E11 down
  0.867 -> 0.688 -> 0.604. The standard's r/d = 5.85 is an order of magnitude
  beyond anything we tested in that sweep.

**Scale is ours to choose; the standard fixes only shape.** All eight
dimensions above are ratios once divided by W. We scale the coupon so that
the gage width spans N unit cells, and **N is then a declared
scale-separation parameter** rather than a tacit assumption. This is the
direct answer to the objection that first-order homogenization needs the
macro field to vary slowly relative to the cell size.

Loading: symmetric imposed traction force at both ends, equal in magnitude.
Force control rather than displacement control, because the previous work
established that a clamped face manufactures its own shear at the free-edge
corner, and force control removes that artifact.

## Micro geometry: rotated elliptical hole

Holes at the **micro scale only**. The macro coupon is solid. This follows
Guo et al. (arXiv:2405.00437), whose microstructure is a perforated sheet
while their macro problem is a plain plate.

Note what we cannot borrow from Guo. Their macro geometry is trivial (a 2x4
element plate) because their *physics* is not: they use second-order
homogenization (F-bar and its gradient G-bar) on a material that buckles, and
that is what gives them a non-uniform macro field. We are first-order on a
stable material, and we verified empirically that this combination gives a
plain bar a perfectly uniform state (g12 ~ 1e-8 under force control). So our
macro non-uniformity has to come from macro geometry. That choice is ours and
must be argued as ours.

**The ellipse is rotated relative to the cell axes.** An axis-aligned
elliptical hole yields orthotropy, which is a named symmetry class. The
paper's contribution is a *group-free* anisotropic construction, so the
microstructure should have no exploitable symmetry group at all. A rotated
ellipse delivers that. The previous D4-symmetric RVE never exercised the
claim the paper is built on.

Free parameters to fix in stage 00: ellipse aspect ratio a/b, porosity (hole
area fraction), rotation angle, and N (cells across the gage width).

### Why the rotated ellipse is structural to the design, not decorative

There is an objection to the coupon that has to be met head-on, because it is
the strongest one against it. With r/d = 5.85 and D/d = 1.46 the shoulder's
concentration factor is close to 1.0. A standardised coupon is *designed* to
produce a uniform uniaxial state in the gage section, since that is its
purpose as a measuring instrument. So the property that makes it safe for
first-order homogenization is the same property that threatens to make it
uninteresting as a structural problem, which is how the plain bar failed.

The rotated ellipse is what answers this. **A rotated elliptical hole gives
the RVE shear-normal coupling**, so a purely axial macro load produces a macro
state with genuine shear. All three strain components are exercised, and the
coupon visibly skews under symmetric tension. The richness comes from the
material rather than from a stress concentration.

This aligns three things that would otherwise pull apart:

- It matches Guo, where the complexity lives in the material and the macro
  problem is deliberately trivial.
- It is exactly the group-free anisotropy the paper claims to handle, so the
  demonstration exercises the actual contribution.
- The gentle fillet stops being a compromise and becomes a virtue: the macro
  state is rich because of the material and safe because of the geometry.

And on why FE-squared at all if the field is smooth: FE-squared tests the
*coupled* solve, the macro Newton iteration consuming the surrogate's tangent.
That is where a non-symmetric or non-positive-definite tangent (the
uncertified tiers) fails. The test is genuine on a mild field.

### Consistency requirements between RVE and FE-squared

Checked before building, since these are what make the whole thing hang
together rather than merely each part being defensible alone.

1. **Cost of the nested-FOM reference, and NO mirror symmetry.** The
   reference needs nonlinear RVE solves at every macro Gauss point at every
   macro Newton iteration, and the coupon is slender (LO/W = 12.7).

   An earlier version of this document claimed a quarter model by symmetry.
   **That is wrong, and the reason matters.** The material is group-free
   anisotropic: reflecting about the long axis maps the ellipse from +30 deg
   to -30 deg, which is not the same material, so the solution is symmetric
   about neither axis. What does survive is POINT symmetry (180 deg), since
   an ellipse is centrosymmetric and so is the loading -- that gives a HALF
   model, not a quarter, and needs multi-point constraints. This is a direct
   consequence of choosing a group-free material, i.e. of the paper's own
   contribution.

   Remaining mitigations: the grip region carries a near-uniform field and can
   be meshed coarsely, and the half-model point symmetry is available if the
   cost demands it. The pre-pass itself is cheap either way, having no nested
   RVE, so it runs on the full coupon.
2. **The RVE must survive the envelope.** Porosity and aspect ratio must be
   chosen so the RVE does not reach pore closure or ligament lock-up anywhere
   inside envelope-plus-margin. Order matters: the pre-pass is cheap and comes
   first, then the RVE is confirmed against it.
   If it does not survive, the microstructure is adjusted, not the coupon.
   That is the opposite of what went wrong before, and the distinction is the
   whole point: previously the *macro geometry* was bent to fit the RVE's
   limits, which made every macro dimension a fitted parameter. Porosity is a
   material design choice that is legitimately ours, and the macro problem
   stays fixed by the standard.
3. **Scale separation is a number, not an assumption.** N cells across the
   gage width, declared. At N = 10 the cell is 1.3 mm against a 13 mm gage.
4. **Plane strain, decided.** The existing machinery is plane strain and is
   already validated, at both the homogenization and the tangent; introducing
   plane stress would mean new unvalidated code paths on both for no gain to
   the message. Two things follow, and both are handled rather than assumed.

   *How it is declared.* The body is **prismatic/extruded at both scales**,
   not a thin plastic coupon. We do not claim to simulate an ASTM tensile
   test; we take the D638 Type I **in-plane profile** as a benchmark geometry
   treated in plane strain, with the microstructure a cell perforated by
   elliptical-*cylinder* holes, likewise extruded. Thickness T is dropped from
   the specification entirely, since only the in-plane profile is used. The
   objection a reviewer can raise (the D638 dimensions target a thin sheet, so
   plane strain is inconsistent with the standard's intent) is presentational
   rather than mechanical, and it is defused by not claiming the physical
   test. The resulting object -- an extruded perforated prism -- is exactly
   what the metamaterial homogenization literature analyses.

   *What it costs, quantitatively.* Plane strain **tightens** the transverse
   budget, which was the binding constraint in the previous attempt (E22
   reaching pore closure). Under uniaxial tension with free lateral edges, in
   the linear limit:

   | | transverse ratio E22/E11 |
   |---|---|
   | plane stress | -nu |
   | plane strain | -nu/(1-nu) |

   because ez = 0 forces sz = nu*sx, so the whole Poisson contraction is
   concentrated in-plane. At nu = 0.3 that is 0.43 against 0.30 (43% more
   contraction); at nu = 0.4, 0.67 against 0.40 (67% more).

   This is a number to respect when sizing porosity in stage 00, not a reason
   to change formulation. A competing effect helps: the effective Poisson
   ratio of a *porous* material is typically lower than the matrix's, since
   pores accommodate contraction. The two partly cancel, and C0 from stage 00
   gives the exact effective figure, which is what stage 01 then propagates.

   *Why it is not binding here.* The factor multiplies a quantity we now
   choose, and the previous attempt was constrained only because its box was
   absurd: E11 up to 2.0, i.e. 200% strain, a range built to impress rather
   than to serve a problem. Resulting E22 in the linear limit:

   | E11 target | nu_eff=0.25 | 0.30 | 0.35 | 0.40 |
   |---|---|---|---|---|
   | 0.10 | -0.033 | -0.043 | -0.054 | -0.067 |
   | 0.20 | -0.067 | -0.086 | -0.108 | -0.133 |
   | 0.30 | -0.100 | -0.129 | -0.162 | -0.200 |

   At 10% axial strain the transverse strain lands between -0.03 and -0.07,
   comfortable even against the *old* cell's -0.1 pore-closure limit, before
   any of the new design freedoms are spent. At 20-30% it depends on nu_eff
   and on the new cell's own limit. That is a design window with three
   upstream knobs, all free: load level (the largest), porosity (raising it
   lowers nu_eff *and* eases pore closure), and ellipse orientation (a hole
   elongated transverse to the load has more travel before closing). The -0.1
   figure is a historical reference from the old cell, not a constant.

### Guard: the load level must not creep

Recorded explicitly because it is the failure mode of the previous attempt
and the kind of thing that gets forgotten halfway through.

The risk to this design is not plane strain. It is the target load drifting
upward in order to print a larger strain number. The load level must be
derived from where first-order homogenization remains defensible, which is
the scope constraint the design is built around, and never from how much
deformation can be shown off.

Note that plane strain and that scope constraint push the same way: both call
for moderate deformation. Plane strain is aligned with the design rather than
fighting it.

One genuine check remains open, to be measured in stage 00 rather than
assumed: the **matrix** Poisson ratio. A nearly incompressible matrix in
plane strain can suffer volumetric locking in the micro solve. At the
effective level this is not a concern, since a porous material has a low
effective bulk modulus, but the matrix level has to be looked at.

## Pipeline

Each stage consumes only the stage before it. Nothing downstream may reach
back and change an upstream choice; that is the discipline being enforced.

| Stage | Produces |
|---|---|
| `00_rve/` | RVE geometry and mesh; effective zero-strain tangent C0 from three linear solves; usable-range check on aspect ratio and porosity |
| `01_macro_prepass/` | Coupon solved with a cheap effective material calibrated to C0; the full cloud of (E11, E22, g12) over every Gauss point and every load step |
| `02_sampling/` | Training domain derived from that cloud plus margin; the sample set; the held-out test set; the explicitly out-of-envelope probe set for 1b |
| `03_data/` | Nonlinear RVE solves at every sample (warm-started); (E -> S, W) pairs plus microscale snapshots |
| `04_training/` | The model tiers trained on that data |
| `05_validation/` | In-envelope accuracy; out-of-envelope behaviour; As'ad-style stability and consistency tests |
| `06_fe2/` | FE-squared runs per tier, against the nested-FOM reference |

## How the training ranges get decided

This is the part that was previously done by hand, and it is the part that
must not be. The ranges are an **output of stages 00-01**, not a choice.

1. **C0 for free.** Three linear RVE solves, one per unit macro-strain
   direction, give the effective zero-strain tangent exactly. No nonlinear
   solves, no fitting.
2. **Cheap pre-pass.** Solve the coupon with a simple effective material whose
   small-strain tangent matches C0, at the target load, and record
   (E11, E22, g12) at every Gauss point at every load step. This costs
   ordinary FE, with no nested RVE anywhere, so the envelope is known before
   any expensive data exists. This is the step whose absence caused the
   previous mess.
3. **Domain from the cloud, with margin.** The recorded cloud is not a box; it
   is a bundle of paths through strain space. The training domain is its
   bounding region inflated by a margin (30-50%), for two reasons that are
   both real: macro Newton *iterates* overshoot the converged path and must
   still be evaluable, and the pre-pass material is not the true material, so
   the cloud itself is only approximate.
4. **A STRUCTURED GRID, not a Sobol point cloud, and not the bounding box.**
   An earlier version of this document called for Sobol sampling on the
   grounds that the RVE is path-independent, so training the constitutive
   surrogates needs a point cloud of (E -> S, W) pairs rather than loading
   paths. That part is true, but it is not sufficient: the *hyperreduction*
   machinery imposes structure that Sobol would break. Three independent
   requirements, all pointing the same way:

   a. **Graph-Laplacian regularization of the adaptive weights.** The MAW-ECM
      weight fields are obtained by solving a joint optimization over all
      sampled states with a Laplacian term coupling NEIGHBOURING states, which
      is what makes the weight field smooth enough to regress. Hernandez
      demonstrates the accuracy degradation without it (alpha_G = 0 against
      10^4). The samples therefore need meaningful, uniformly spaced
      neighbourhood relations. A Cartesian grid supplies them naturally, and
      also permits direct graphical inspection of the weight fields, which is
      the second reason he gives for restricting to a rectangle.

   b. **Identification of the decoder's latent coordinates.** The factorized
      decoder is `d_red = Phi_M A_M q_M + Phi_S N_slave(q_M)`, with
      `q_M = T_m Phi_ROM^T d_red` and `T_m` identified input-informed. For
      that identification to be well posed the snapshots must cover a manifold
      densely and uniformly INDEXED BY mu. Latent dimension follows the
      parameter dimension: **r_D = 3** here (E11, E22, gamma12), against 2 in
      Hernandez's metamaterial benchmark.

   c. **Conditioning of the affine mu -> q_p least-squares map.** See below;
      this is what distinguishes D-HPROM-ANN from HPROM-ANN, and it needs mu
      to span all three directions.

   **Grid in coordinates ALIGNED WITH THE CLOUD, not over the box.** Measured
   from the pre-pass cloud: the bounding box is only **9.1% occupied**, so a
   grid over it would waste ~91% of its samples. The cloud is a thin bundle
   about a dominant ray, `E22 = -0.4686 E11` and `g12 = -0.2921 E11`, which
   matches C0's uniform-state prediction (-0.4718, -0.3116) to 0.7% and 6.3%.
   So the grid runs over

       s   = E11                    dense, the load parameter
       d22 = E22 - ray22(s)         thin: measured +-0.017
       d12 = g12 - ray12(s)         fat:  measured +-0.099

   which is dense where the problem lives while preserving neighbourhood
   relations, the map being a linear shear of bounded distortion. It also
   fixes the previous design's defect by construction: there, shear sat at
   only five discrete levels with unsampled gaps between the planes.

   Snapshots for the reduced-order side come free regardless: each RVE solve
   internally ramps to its target and emits microscale displacement fields
   along the way.

5. **The affine mu -> q_p map is the D-HPROM-ANN's structural assumption, and
   it must be tested early.** In the code the object is
   `qp_init_mu_affine`, and `q_p = [mu, 1] @ b_aff`. The same object plays two
   different roles:

   | | role of the affine mu -> q_p map |
   |---|---|
   | HPROM-ANN (iterative) | INITIALIZER; Newton then solves the projected equilibrium for q_p |
   | D-HPROM-ANN (direct) | the ANSWER; there is no equilibrium solve |

   So D-HPROM-ANN's accuracy is bounded by how affine `q_p` actually is in mu.
   That is a structural assumption of the method, not a data-quantity issue:
   if `q_p` is not near-affine, no amount of data removes the error floor --
   and that would be a reportable finding rather than a bug.

   Why it tends to hold: at moderate strain the micro fluctuation responds
   almost linearly to the macro strain, so `q_p ~ A mu` with a small
   correction, and `N_ANN(q_p)` absorbs the nonlinearity in the slave
   coordinates. That is an EMPIRICAL property of the moderate regime, not a
   guarantee.

   Two consequences. First, mu must span all three directions well enough to
   condition the least squares; with the measured spans (E11 0.163, d12 0.198,
   d22 0.034) the design matrix conditions at about 6, acceptable, with d22
   the direction to watch -- and the 40% margin widening it turns out to serve
   this purpose too. Second, **stage 04 measures the affine fit's residual
   before any tier comparison is run**, and measures it under BOTH snapshot
   conventions: POD on the fluctuation (what this project's
   `hprom_solver_rve` does, `u_aff_free + phi_f q`) and POD with the affine
   part retained (Hernandez's, where `q_p` is dominated by a term exactly
   linear in mu and the affine map is structurally better). The convention is
   then chosen from the measured residual rather than assumed.
5. **Warm-started generation.** Each new sample continues from its nearest
   already-solved neighbour instead of ramping from zero. Verified exact
   rather than approximate for this material class (1e-15 relative on both S
   and CC), which is what makes a dense cloud affordable.
6. **Held-out sets, defined up front.** A test set inside the envelope, and a
   separate probe set deliberately outside it for 1b. Declared before
   training, never used to select anything.

The caveat on step 5 carries over and is stated rather than assumed:
path-independence fails wherever the RVE admits several stable branches at one
E, which for a perforated cell means pore buckling under compression. Problem
1 is tension-dominated, so this is safe here; near pore closure, cold starts.

## Engineering discipline

The point of doing this from scratch is not to redo the same work in a new
folder. Time lost to inconsistency is never lost in the stage that produced
it; it is lost three stages later, after something was built on top. So every
stage must be able to **fail loudly** rather than silently hand a bad result
downstream.

**One source of truth for parameters.** A single `config.py` (or equivalent)
that every stage reads. No parameter appears in a function default *and* in a
caller's dict. The previous codebase had exactly that duplication: geometry in
per-runner `GEOM` dicts, mirrored in the runner signatures' own defaults
(`W_gauge=4.0` in both places). That is how two rows of a results table end up
belonging to two different problems, discovered days later.

**Declared units.** The ASTM dimensions are in mm; the previous code used
`FORCE = 1.0e9`, implying Pa. Mixing mm with N/m^2 breaks nothing visibly and
simply produces wrong numbers. One system is declared once and checked.

**Reproducibility.** Fixed seeds for the Sobol/LHS sampling. Every generated
dataset carries a manifest recording the config that produced it, so a
dataset can never be silently paired with the wrong geometry or envelope.

**Boundary conditions on the RVE stated, not implied.** **Affine Dirichlet**
(linear displacement), which is what the existing machinery implements
(`core/fom_solver_rve.py`, "analytical affine Dirichlet values from
Green-Lagrange strain"); periodic BCs are not implemented anywhere in `core/`.

Why keep affine rather than add periodic. The paper's message concerns
surrogate *admissibility*, not recovery of some true effective property of a
real material. The nested-FOM reference and every surrogate share the same
RVE definition, so the comparison is exact and internally consistent under
either BC. Against that, implementing periodic BCs means re-deriving and
re-validating the **analytic tangent under constraints** -- the most delicate
piece of machinery here, and the thing that makes the FOM reference
affordable at all (1 RVE solve per Gauss point instead of 7). That is new,
unvalidated code in the critical path for no gain to the message.

What it costs, declared rather than hidden. There is a rigorous ordering:

    C_Neumann  <=  C_periodic  <=  C_affine-Dirichlet

so **affine Dirichlet over-stiffens**, and on a single unit cell the effect
can be appreciable; periodic converges fastest to the true property. This is
measured rather than left as an open objection: stage 00 compares C0 from
1x1, 3x3 and 5x5 cell blocks under the same affine BCs. These are **linear
solves only**, so the study is nearly free, and if C0 has converged by 3x3 the
boundary-layer effect is small and gets reported as a number.

### Per-stage acceptance tests

No stage is considered done, and nothing downstream may start, until its own
tests pass.

**00_rve**
- C0 symmetric, positive definite.
- C0 components 13 and 23 nonzero. If they vanish, the rotation is doing
  nothing and the cell is not group-free -- the paper's central claim would go
  untested. This is the single most important check in the stage.
- C0 converged under mesh refinement (refine, C0 changes below tolerance).
- C0 converged under **cell count**: 1x1 vs 3x3 vs 5x5 under affine BCs.
  Note these will *not* agree exactly, unlike under periodic BCs -- affine
  Dirichlet over-stiffens through a boundary layer, so the convergence rate
  with cell count *is* the result. Report the residual over-stiffening as a
  number. Linear solves only, so this is cheap.
- Matrix Poisson ratio inspected for plane-strain volumetric locking.
- Usable tensile range measured: where pore closure or ligament lock-up
  begins.

**01_macro_prepass**
- Reaction force equals applied force (global equilibrium).
- Solution respects the problem's own symmetry.
- Mesh-converged strain cloud. Cheap here, since there is no nested RVE, and
  reusable as the macro mesh justification later.

**02_sampling**
- Sample set covers the derived domain (no unsampled voids, the specific
  defect of the previous design's five discrete shear planes).
- Grid is structured with uniform neighbourhood relations, since the MAW-ECM
  weight optimization couples neighbours through a graph Laplacian.
- The grid, expressed back in (E11, E22, g12), CONTAINS the measured pre-pass
  cloud. A grid aligned to a fitted ray could otherwise miss cloud points that
  deviate from it.
- Design matrix `[mu, 1]` for the affine mu -> q_p fit is well conditioned in
  all three directions, d22 being the thin one.
- Test set and out-of-envelope probe set fixed and recorded before any
  training runs.

**04_training** (additional, before any tier comparison)
- Residual of the affine `q_p = [mu, 1] b_aff` fit, measured under BOTH
  snapshot conventions (fluctuation-only and affine-retained). This bounds
  D-HPROM-ANN independently of how much data exists, so it is measured before
  effort is spent on the comparison.
- Latent dimension r_D = 3, matching the parameter dimension.

**03_data**
- Warm-started solves match cold solves to roundoff on a spot-check subset.
- Stress consistent with the energy: S recovered from W by differentiation
  agrees with the S written out.
- Analytic tangent agrees with a finite-difference tangent on a spot-check
  subset.

**04_training / 05_validation**
- In-envelope accuracy per tier against held-out data.
- Out-of-envelope behaviour per tier.
- Static zero-load consistency: a body with no applied load must not deform.
  This is As'ad's test in its static form and it is the stability half of 1b.

**06_fe2**
- Patch test first: a uniform macro strain must reproduce the RVE's own
  answer exactly. If this fails, nothing else in the stage means anything.
- Then the tier runs, against the nested-FOM reference.

## Sources

- ASTM D638-14, dimension table read directly from the standard text.
- Guo, Kouznetsova, Geers, Veroy, Rokos, "Reduced-order modeling for
  second-order computational homogenization with applications to
  geometrically parameterized elastomeric metamaterials", arXiv:2405.00437.
  Their recipe: 2000 snapshots from 100 random uniform samples inside bounds
  derived from the intended loading, each ramped in 20 steps. Their stated
  condition: the reduced model approximates full computational homogenization
  well "provided that the training data is representative for the problem at
  hand".
- As'ad and Farhat, for the structure of the failure demonstrations
  (in-domain vs out-of-domain separation; stability and consistency tests).
