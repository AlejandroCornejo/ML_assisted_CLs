# FOM energy-tangent audit

The reported matrix is the central finite-difference estimate of
`D = d²W/de² = dS/de`, not the derivative of Cauchy stress.

Finite-difference step: `0.005`.

| State | e=[E11,E22,gamma12] | eigenvalues of D [GPa] | SPD? |
|---|---:|---:|:---:|
| stage10_mixed | [0.885714, 0.8, 0.0428571] | [-0.0648005, -0.0310749, 0.128034] | no |
