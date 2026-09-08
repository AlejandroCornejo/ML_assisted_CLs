#!/usr/bin/env python3
"""Single source of truth for every parameter in this study.

Every stage imports from here. No parameter may appear both in a function
default and in a caller's dict -- that duplication is what let the previous
codebase end up with two table rows belonging to two different problems
(geometry lived in per-runner GEOM dicts *and* mirrored in the runner
signatures' own defaults).

If you need a value, import it. Do not retype it.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# UNITS
# ---------------------------------------------------------------------------
# SI throughout: metres, newtons, pascals. Chosen so the validated material
# block (E in Pa) carries over verbatim with no conversion anywhere. The ASTM
# profile is therefore expressed in METRES, not millimetres.
UNITS = "SI (m, N, Pa)"

# ---------------------------------------------------------------------------
# MICRO: unit cell with a rotated elliptical hole
# ---------------------------------------------------------------------------
# The cell lives in its OWN coordinates, side 2 centred at the origin, exactly
# as the previously validated rve_geometry.mdpa does. This is legitimate and
# not a unit inconsistency: a hyperelastic material has no intrinsic length
# scale, so the homogenized response S(E) is invariant under a uniform
# rescaling of the cell. Only the SHAPE parameters below affect it. The cell's
# physical size enters solely through the scale-separation declaration.
CELL_SIDE = 2.0
CELL_AREA = CELL_SIDE ** 2

# Rotated ellipse. Starting values, to be revised from stage-00 measurements
# rather than from opinion.
POROSITY = 0.20            # hole area / cell area
ELLIPSE_ASPECT = 2.0       # a/b, a the major semi-axis
ELLIPSE_ANGLE_DEG = 30.0   # major axis vs +x

# 30 degrees is deliberately not 0/45/90. At those angles the rotated ellipse
# retains a mirror plane of the square cell and the effective response stays
# orthotropic, which would leave the paper's group-free claim untested. Note
# that the configuration is still centrosymmetric, which is harmless: every
# elastic tensor is automatically even in the strain, so point symmetry
# imposes no restriction on C0.


def ellipse_semi_axes(porosity=POROSITY, aspect=ELLIPSE_ASPECT, cell_area=CELL_AREA):
    """(a, b) with pi*a*b = porosity*cell_area and a/b = aspect."""
    ab = porosity * cell_area / np.pi
    b = np.sqrt(ab / aspect)
    return aspect * b, b


def ellipse_bounding_half_extents(porosity=POROSITY, aspect=ELLIPSE_ASPECT,
                                  angle_deg=ELLIPSE_ANGLE_DEG, cell_area=CELL_AREA):
    """Half-width and half-height of the rotated ellipse's bounding box.
    Both must stay below CELL_SIDE/2 or the hole breaches the cell boundary."""
    a, b = ellipse_semi_axes(porosity, aspect, cell_area)
    t = np.radians(angle_deg)
    return (np.hypot(a * np.cos(t), b * np.sin(t)),
            np.hypot(a * np.sin(t), b * np.cos(t)))


# ---------------------------------------------------------------------------
# MICRO: deployed mesh
# ---------------------------------------------------------------------------
# Chosen on LOCAL FIELD accuracy, not on the homogenized average. Measured at
# E11 = 0.20 on the true uniaxial path, relative to a 3095-element reference:
#
#   elements  GPs   max||P||   S11 (average)
#        253  759   1.82e-02   3.5e-04       <- average converged, field not
#       1546 4638   4.86e-03   9.9e-06
#
# The average is converged 52x better than the local maximum at 253 elements,
# because averages converge faster than the fields they average. POD and the
# ECM operate on the FIELD, so the field is the criterion. 467 elements were
# a bad deal (max only 1.82% -> 1.49% for 1.85x the points), so the real
# choice was 253 or 1546.
#
# That 4638 integration points lands at 0.88x the 5260 of Hernandez's MAW-ECM
# metamaterial benchmark is a welcome consequence, not the reason.
MESH_SIZE_FAR = 0.14
MESH_SIZE_HOLE = 0.05

# ---------------------------------------------------------------------------
# MICRO: material (matrix)
# ---------------------------------------------------------------------------
# Reused verbatim from core/StructuralMaterials.json, so the validated
# constitutive path is untouched.
MATRIX_YOUNG = 1.628e9      # Pa
MATRIX_POISSON = 0.4
MATRIX_DENSITY = 7850.0
MATRIX_THICKNESS = 0.05     # cancels out of the homogenized stress
CONSTITUTIVE_LAW = "HyperElasticPlaneStrain2DLaw"
ELEMENT_NAME = "TotalLagrangianElement2D6N"

# Plane strain, decided. K/G = E/(3(1-2nu)) / (E/(2(1+nu))) = 4.67 at nu=0.4,
# far from the K/G >> 1 regime where volumetric locking bites (nu > ~0.49),
# and T6 elements are robust there in any case. Checked, not assumed.

# ---------------------------------------------------------------------------
# MICRO: homogenization convention
# ---------------------------------------------------------------------------
# DIVIDE BY THE CELL AREA, INCLUDING THE VOID.
#
# This differs deliberately from the inherited default. The solver's
# GetReferenceIntegrationMeasureFromMesh returns A_ref = sum_e Area_e, i.e.
# the SOLID area only, which for the previous 19.63%-porous cell over-reported
# macro stress by a factor 1.2438 (4.0 / 3.2159, measured). That was
# internally consistent -- reference and surrogates shared it, so every
# published comparison stands -- but it makes the macro problem solve with a
# material ~24% stiffer than the porous cell's true effective material.
#
# Macro stress is force per unit MACRO area, which must include the void, so
# the cell area is the physically correct measure. Crucially this is a
# CONFIGURATION change, not a code change: hom_reference_measure is already a
# solver parameter and is propagated into final_state, so the analytic tangent
# divides by the same number automatically. No new code in the critical path.
HOM_REFERENCE_MEASURE = CELL_AREA

# ---------------------------------------------------------------------------
# MACRO: ASTM D638 Type I in-plane profile, in metres
# ---------------------------------------------------------------------------
# Read directly from the D638-14 standard text, not from secondary sources.
# Thickness T is deliberately ABSENT: only the in-plane profile is used, and
# the body is declared prismatic in plane strain rather than a thin coupon.
COUPON_W_GAUGE = 13.0e-3     # W, width of narrow section
COUPON_L_GAUGE = 57.0e-3     # L, length of narrow section
COUPON_W_GRIP = 19.0e-3      # WO, width overall
COUPON_L_TOTAL = 165.0e-3    # LO, length overall
COUPON_GAGE_LENGTH = 50.0e-3  # G
COUPON_GRIP_SEP = 115.0e-3   # D, distance between grips
COUPON_FILLET_R = 76.0e-3    # R, radius of fillet

# Derived, and the reason this geometry was chosen: the standard proportions
# make the shoulder a weak concentrator (a coupon must fail in the gage, where
# the extensometer sits), which is exactly the gentle macro gradient
# first-order homogenization wants.
COUPON_D_OVER_D = COUPON_W_GRIP / COUPON_W_GAUGE      # 1.4615
COUPON_R_OVER_D = COUPON_FILLET_R / COUPON_W_GAUGE    # 5.8462

# ---------------------------------------------------------------------------
# SCALE SEPARATION
# ---------------------------------------------------------------------------
# Declared as a number rather than left as a tacit assumption. This is the
# only place the micro and macro length scales meet.
CELLS_ACROSS_GAUGE = 10
PHYSICAL_CELL_SIZE = COUPON_W_GAUGE / CELLS_ACROSS_GAUGE   # m

# ---------------------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------------------
# Ranges are NOT set here. They are an output of stages 00-01: C0 from three
# linear RVE solves, then the cheap macro pre-pass records the strain cloud
# the coupon actually visits, and the training domain is that cloud inflated
# by ENVELOPE_MARGIN. Anything that hardcodes a strain range in this file is
# a bug in the method, not a convenience.
# DO NOT trust this number until stage 01 measures it. The original 0.40 was
# a guess, and measuring showed a guess here is dangerous.
#
# The pre-pass material (St Venant-Kirchhoff with C0) is far too stiff at
# finite strain, because the real perforated cell SOFTENS as its ligaments
# reorient. Measured against the true uniaxial path: S11 over-predicted by
# +19.7% at E11 = 0.10, +40.8% at 0.20, +109.6% at 0.50. Inverted for FORCE
# control, that means the strain is under-predicted by 28% / 76% / 236% at
# those levels -- so a 40% margin would have undersized the envelope by a
# factor of 1.8 to 3.4 and stage 03 would have generated data in the wrong
# region.
#
# What is well predicted is the KINEMATIC ratios: gamma12/E11 to within 2.5%
# and E22/E11 to within 6-31%. The stiffness is wrong, the directions are not.
# Hence the pre-pass runs under DISPLACEMENT control, where the strain field is
# set kinematically and a too-stiff material gives the same strains with
# higher forces. The FE^2 runs stay force-controlled; the force is calibrated
# afterwards against the real model.
#
# MEASURED (stage 01). The obvious test would have been a null test: scaling
# C by a scalar leaves the strain field EXACTLY unchanged, the equilibrium
# equations being homogeneous in C. What genuinely moves the field under
# displacement control is the material's NONLINEARITY, since a softening
# material lets the more-strained gauge take a larger share of the elongation.
#
# So the probe was SVK-C0 against a variant scaled by phi(||E||) calibrated to
# the measured true RVE uniaxial response. Two materials differing by up to
# 110% in stress gave strain clouds agreeing as follows:
#
#   E11 max  0.04%    E22 min  0.57%    g12 min  3.15%    g12 max  5.11%
#
# i.e. the envelope EXTREMES are material-insensitive to ~5%, and the required
# end displacement differed by only 2.9% (20.02 vs 19.44 mm).
#
# 0.40 therefore covers the measured 5% with generous room for the residual
# the probe cannot capture -- it holds C0's structure fixed and only scales it,
# whereas the true material's anisotropy also evolves with strain.
#
# Newton overshoot is deliberately NOT folded into this number. It is a
# property of the FE^2 macro solve driven by the surrogate, not of the
# pre-pass, so it cannot be measured here. Stage 06 instead INSTRUMENTS the
# FE^2 runs to flag any Gauss-point query falling outside the trained domain,
# which is a runtime check and strictly more reliable than a margin guess.
ENVELOPE_MARGIN = 0.40
SAMPLING_SEED = 20260903   # fixed, so datasets are reproducible


def summary():
    a, b = ellipse_semi_axes()
    hw, hh = ellipse_bounding_half_extents()
    half = CELL_SIDE / 2.0
    K = MATRIX_YOUNG / (3.0 * (1.0 - 2.0 * MATRIX_POISSON))
    G = MATRIX_YOUNG / (2.0 * (1.0 + MATRIX_POISSON))
    return "\n".join([
        f"units                    {UNITS}",
        f"cell side / area         {CELL_SIDE} / {CELL_AREA}",
        f"porosity                 {POROSITY * 100:.2f} %",
        f"ellipse a, b             {a:.6f}, {b:.6f}  (a/b = {ELLIPSE_ASPECT})",
        f"ellipse angle            {ELLIPSE_ANGLE_DEG} deg",
        f"bounding half-extents    {hw:.4f} x {hh:.4f}  (cell half = {half})",
        f"  ligament margin        {half - hw:.4f} (x), {half - hh:.4f} (y)",
        f"matrix E, nu             {MATRIX_YOUNG:.4e} Pa, {MATRIX_POISSON}",
        f"  K/G                    {K / G:.3f}  (locking regime is K/G >> 1)",
        f"hom reference measure    {HOM_REFERENCE_MEASURE} (cell area, void included)",
        f"coupon D/d, r/d          {COUPON_D_OVER_D:.4f}, {COUPON_R_OVER_D:.4f}",
        f"coupon gauge W x L       {COUPON_W_GAUGE * 1e3:.1f} x {COUPON_L_GAUGE * 1e3:.1f} mm",
        f"scale separation         {CELLS_ACROSS_GAUGE} cells across gauge, "
        f"cell = {PHYSICAL_CELL_SIZE * 1e3:.3f} mm",
        f"mesh sizes (far/hole)    {MESH_SIZE_FAR} / {MESH_SIZE_HOLE}",
        "envelope margin          set by stage 01, from measurement"
        if ENVELOPE_MARGIN is None else
        f"envelope margin          {ENVELOPE_MARGIN * 100:.0f} %",
        f"sampling seed            {SAMPLING_SEED}",
    ])


if __name__ == "__main__":
    print(summary())
    hw, hh = ellipse_bounding_half_extents()
    assert max(hw, hh) < CELL_SIDE / 2.0, "ellipse breaches the cell boundary"
    print("\nCONFIG_OK")
