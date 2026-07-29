# FOM energy-tangent audit

The reported matrix is the central finite-difference estimate of
`D = d²W/de² = dS/de`, not the derivative of Cauchy stress.

Finite-difference step: `0.01`.

| State | e=[E11,E22,gamma12] | eigenvalues of D [GPa] | SPD? |
|---|---:|---:|:---:|
| reference | [0, 0, 0] | [0.50167, 1.01928, 3.00555] | yes |
| compressed_shear | [-0.1, -0.1, 0.1] | [1.14216, 2.38524, 6.9836] | yes |
| moderate_mixed | [0.5, 0.5, 0.05] | [-0.017159, -0.00731696, 0.307954] | no |
| large_biaxial | [1, 1, 0] | [-0.0695258, -0.033408, 0.0918331] | no |
| extreme_biaxial_shear | [2, 2, 0.1] | [-0.0590125, -0.0285127, 0.0174626] | no |
| stage10_mixed | [0.885714, 0.8, 0.0428571] | [-0.0648006, -0.0310749, 0.128038] | no |
