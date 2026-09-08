# Stage 01 results: the macro pre-pass

Coupon: ASTM D638 Type I in-plane profile, plane strain, prismatic body.
Material: St Venant-Kirchhoff with the measured periodic C0.
Control: DISPLACEMENT, symmetric (-delta/2 left, +delta/2 right), end-face y
left free so no face is clamped.

## Profile, and the cotes closing

| | mm |
|---|---|
| gauge half-width W/2 | 6.5000 |
| grip half-width WO/2 | 9.5000 |
| fillet rise | 3.0000 |
| fillet run dx = sqrt(R^2-(R-rise)^2) | 21.1424 |
| gauge half-length | 28.5000 |
| fillet ends at | 49.6424 |
| specimen half-length LO/2 | 82.5000 |
| fillet end slope | 16.15 deg |

The fillet is tangent to the NARROW section and meets the wide section at a
16.15 deg kink, which follows from the standard specifying a single R -- being
tangent at both ends would need a two-arc S-curve. The kink sits in the wide
section, where stress is lower by the width ratio 13/19 = 0.68, and far from
the gauge.

The cotes close against the standard's own independent dimensions, which is
the check that the profile is the standard's and not an invention: the grips
clamp at D/2 = 57.5, i.e. inside the straight grip region (57.5 > 49.64), and
the mesh area matches the analytic profile area to 8.4e-05 relative.

Mesh: 4233 nodes, 2016 elements, 6048 Gauss points.

## Calibration of end displacement to gauge strain

The naive `delta = (lambda-1) L_gauge` is wrong by a factor of ~2.4: the
imposed displacement is shared over the whole 165 mm specimen, not just the
57 mm gauge, and the wider grips strain less. Weighting each region by
w_gauge/w(x) as uniaxial equilibrium requires gives an effective length of
~137 mm, which came out 4.5% low against the measured value -- good enough as
a starting point for a secant iteration on the MEDIAN gauge E11.

Converged in 3 iterations: **delta = 20.0166 mm gives gauge E11 median
0.15000** exactly.

## The strain cloud at gauge E11 = 0.15

60480 (Gauss point, load step) samples.

| | min | p1 | p99 | max |
|---|---|---|---|---|
| E11 | +0.00566 | +0.00965 | +0.15046 | **+0.16212** |
| E22 | **-0.08031** | -0.07114 | -0.00456 | -0.00200 |
| g12 | **-0.12639** | -0.07674 | +0.01900 | **+0.06625** |

Two readings:

1. **E11 concentration factor is only 1.08** (0.16212 / 0.15). The standard's
   gentle fillet does exactly what was predicted when the geometry was chosen:
   Kt ~ 1. The coupon really is the smooth, defensible object it was picked to
   be.
2. **g12 reaches -0.126, i.e. 0.84x the gauge E11**, far above the 0.31 the
   RVE's uniform state gives. The shear CONCENTRATES in the fillet region,
   the geometry adding to the material's own coupling. The envelope must cover
   g12 to about +-0.13, not the +-0.047 a uniform gauge state would suggest.

## Why displacement control, measured

SVK-C0 is far too stiff at finite strain, because the real perforated cell
softens as its ligaments reorient. Against the measured true uniaxial path:

| E11 | S11 true | S11 SVK-C0 | error |
|---|---|---|---|
| 0.10 | 1.012e8 | 1.211e8 | +19.7% |
| 0.20 | 1.720e8 | 2.422e8 | +40.8% |
| 0.50 | 2.889e8 | 6.054e8 | +109.6% |

Inverted for FORCE control that is catastrophic -- at the load SVK says gives
E11 = 0.10 / 0.20 / 0.30, the true material gives 0.128 / 0.351 / 1.007, i.e.
strain under-predicted by 28% / 76% / 236%. A force-controlled pre-pass with
this material would have undersized the envelope by up to 3.4x and stage 03
would have generated data in the wrong region.

What SVK-C0 DOES get right is the kinematic ratios: gamma12/E11 to within
2.5%, E22/E11 to within 6-31%. The stiffness is wrong, the directions are not.
Hence displacement control, where the strain field is set kinematically and a
too-stiff material merely reports higher forces.

The FE^2 runs stay force-controlled; the force is calibrated later against the
real model, and the envelope measured here already covers it.

## Envelope margin, measured rather than guessed

The obvious sensitivity test is a NULL test: scaling C by a scalar leaves the
strain field exactly unchanged, the equilibrium equations being homogeneous in
C. What genuinely moves the field under displacement control is the material's
NONLINEARITY, since a softening material lets the more-strained gauge take a
larger share of the elongation.

So the probe was SVK-C0 against a variant scaled by phi(||E||) interpolated
from the measured true uniaxial response (a probe, not a constitutive model --
it has no potential). Result:

| | SVK | softening probe | rel |
|---|---|---|---|
| delta | 20.0166 mm | 19.4366 mm | 2.9% |
| E11 max | +0.16212 | +0.16219 | 0.04% |
| E22 min | -0.08031 | -0.07984 | 0.57% |
| g12 min | -0.12639 | -0.12240 | 3.15% |
| g12 max | +0.06625 | +0.06287 | **5.11%** |

**Two materials differing by up to 110% in stress give strain clouds agreeing
to 5.11% worst case**, and the envelope EXTREMES agree to 0.04-5.11%. The
displacement-control decision is validated quantitatively.

`ENVELOPE_MARGIN = 0.40` therefore covers the measured 5% with generous room
for the residual the probe cannot capture: it holds C0's structure fixed and
only scales it, whereas the true material's anisotropy also evolves with
strain.

Newton overshoot is deliberately NOT folded into that number. It is a property
of the FE^2 macro solve driven by the surrogate, not of the pre-pass, so it
cannot be measured here. Stage 06 instead instruments the FE^2 runs to flag
any Gauss-point query outside the trained domain -- a runtime check, strictly
more reliable than a margin guess.

## Load level: gauge E11 = 0.15

Recommended and adopted. With the 40% margin the training domain is about
`E11 in [0, 0.23]`, `E22 in [-0.11, 0]`, `g12 in [-0.18, +0.09]`.

Both criteria are comfortable:

* **RVE survival**: the cell holds to E11 >= 1.70, so this uses 13% of its
  range. The RVE is not the constraint, which is the healthy situation and the
  opposite of the previous attempt.
* **Scale separation**: E11 goes from 0.15 in the gauge to 0.162 at the
  concentration over ~20 mm of fillet, against a 1.3 mm cell. The macro field
  varies very slowly relative to the cell.

## Macro mesh convergence of the cloud

| mesh | elems | GPs | delta (mm) | E11max | E22min | g12min | g12max |
|---|---|---|---|---|---|---|---|
| W/3 | 424 | 1272 | 20.0125 | 0.15988 | -0.07806 | -0.11807 | +0.05272 |
| W/4 | 690 | 2070 | 20.0161 | 0.16072 | -0.07874 | -0.12134 | +0.05759 |
| W/6 | 1232 | 3696 | 20.0163 | 0.16201 | -0.07959 | -0.12494 | +0.06332 |
| W/8 | 2016 | 6048 | 20.0166 | 0.16212 | -0.08031 | -0.12639 | +0.06625 |
| W/12 | 3674 | 11022 | 20.0139 | 0.16270 | -0.08076 | -0.12783 | +0.06869 |

Judged as absolute error over the ENVELOPE SPAN, which is the metric that
matters when the bounds become a sampling domain:

| mesh | GPs | E11max | E22min | g12min | g12max | worst | absorbed by the 40% margin |
|---|---|---|---|---|---|---|---|
| W/3 | 1272 | 1.73% | 3.34% | 4.97% | 8.13% | 8.13% | 5x |
| W/4 | 2070 | 1.22% | 2.50% | 3.30% | 5.65% | 5.65% | 7x |
| W/6 | 3696 | 0.42% | 1.45% | 1.47% | 2.73% | 2.73% | 15x |
| W/8 | 6048 | 0.36% | 0.56% | 0.73% | 1.24% | **1.24%** | **32x** |

Raw PER-BOUND relative errors are misleading here and should not be used:
E11.min and E22.max come out at 11.6% and 8.8% even at W/8, but those are the
LEAST-strained points, sitting essentially at the origin of the cloud
(E11.min ~ 0.006, E22.max ~ -0.002), so their absolute difference is ~0.0007,
i.e. 0.4% of the envelope span. They fall trivially inside any domain covering
the extremes.

**The envelope is converged at the pre-pass mesh (W/8): 1.24% of span, which
the 40% margin absorbs 32 times over.** g12max is the slowest bound, as shear
was in the RVE mesh study too.

### Consequence for the FE^2 macro mesh

`delta` is converged from **W/4** onward (20.0161 against 20.0139 at W/12, and
non-monotone at the 1e-4 level), so the global force-displacement response is
converged at 690 elements / 2070 Gauss points -- again averages converging
well before fields. Since every macro Gauss point in FE^2 is a full nonlinear
RVE solve, at the measured ~0.6 s per RVE tangent, 3 Newton iterations and 20
load steps on 16 workers:

| macro mesh | GPs | FE^2 reference estimate |
|---|---|---|
| W/4 | 2070 | ~1.3 h |
| W/6 | 3696 | ~2.3 h |
| W/8 | 6048 | ~3.8 h |

Stated with the proper reservation: the FE^2 macro mesh needs its OWN
convergence criterion, on the paper's output quantities rather than on the
envelope, because the surrogate responds differently from SVK-C0. The delta
convergence is a strong hint, not a proof. That study belongs to stage 06.

## Errors found and fixed in this stage

1. **Target-to-displacement conversion** spread the elongation over the 57 mm
   gauge instead of the ~137 mm effective length, so a requested gauge E11 of
   0.15 produced 0.063. Replaced by a secant iteration on the measured value.
2. **The sensitivity test as originally designed was a null test** (scalar
   scaling of C). Replaced by the measured-response softening probe.
3. **The softening probe's consistent tangent diverged Newton.** Diagnosed
   rather than guessed: the tangent is consistent (1e-11 against finite
   differences at small strain) and positive definite, but ASYMMETRIC (7.9e-02
   at mid strain, 1.7e-01 at high) because the rank-1 correction is not
   symmetric -- the signature of the probe having no potential. Switched to
   modified Newton with phi*C, which is symmetric and positive definite;
   Newton's fixed point does not depend on the tangent, so the converged field
   is unchanged.
4. **A quarter model by symmetry is impossible**, corrected in the design
   document. The material is group-free anisotropic, so reflecting about the
   long axis maps the ellipse from +30 to -30 deg -- not the same material.
   Only point symmetry survives, giving a half model at the cost of
   multi-point constraints.
