# Physical response of material B

Diagnostic postprocessing of the 32 original ray targets saved on the
8,961-element check mesh. No new FOM solves or training were performed.

[Figure](physical_response.png) · [Vector PDF](physical_response.pdf)
· [Numerical values](response.csv) · [Source hashes and definitions](response.json).

Panel (a) shows the signed second-Piola axial component Sii against its
Green–Lagrange strain Eii. Transverse normal strain and shear are imposed zero;
transverse stress is not zero. This is not uniaxial-stress loading.
X/Y curves almost overlap in these tests; that alone does not establish isotropy.

Panels (b,c) use the ray parameter t in e(t) = t e_end, where
e = (E11, E22, 2 E12). It is a loading fraction, not time. Endpoints differ
between paths; equal t does not mean equal deformation. All endpoint vectors
are in response.json. Each ray has four saved nonzero targets. The zero
reference is analytically normalized; straight lines are visual guides, not
additional solved increments or a fitted constitutive model.

Von Mises is evaluated **after homogenizing** the full Cauchy tensor:

    sigma_eq = sqrt(3/2 * dev(sigma_bar):dev(sigma_bar)).

Its in-plane block is Fbar Sbar Fbar^T / Jbar, with
Fbar = sqrt(I + 2 E) and Jbar = det Fbar. This agrees with integration of
Pmicro Fmicro^T over the reference solid, divided by Jbar times the full
reference-cell volume, to a worst relative discrepancy of 1.13e-14.

For the underlying compressible 3D Neo-Hookean matrix in plane strain,
F33 = 1 gives P33 = lambda log(Jmicro). Thus

    sigma33_bar = sum(w lambda log(Jmicro)) / (Jbar A0 thickness),
    lambda = Young * nu / ((1 + nu)(1 - 2 nu)).

Here w includes reference-solid area and thickness, A0 is the full cell area,
and Young/nu/thickness are read from the original frozen material config.
This uses the underlying matrix's 3D constitutive extension, not a general
3D extension of the learned in-plane energy. Setting sigma33 to zero would
instead introduce an unjustified plane-stress assumption. The plotted
equivalent is not the average or maximum of microscopic equivalents.

Von Mises is nonnegative, loses the loading sign and ignores hydrostatic
stress; it is a visualization here, not a plastic-yield condition. Neither
these curves nor apparent stress nonlinearity prove hyperelasticity. The
potential structure and independent energy-gradient checks supply that
separate evidence. Nearly linear stress over a limited range is compatible
with hyperelasticity.

Reproduction from the repository root, using a new output folder:

    PYTHONPATH=coupon_fe2_paper/.pydeps coupon_fe2_paper/.venv_fe2/bin/python \
      coupon_fe2_paper/07_material_b/plot_physical_response.py \
      --out coupon_fe2_paper/07_material_b/physical_response_new
