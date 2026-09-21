# Second, stronger exploratory search — user-authorized follow-up

The first bounded audit found no negative Free curvature. The user explicitly
requested a stronger search after discussing its limited optimization budget.
This new post-test experiment is separate; the first results remain intact.

Freeze the following before execution:

- Same three frozen Free checkpoints. No fitting or checkpoint selection.
- Parameterize F=R(theta) diag(lambda1,lambda2) R(theta)^T, with both principal
  stretches in [0.55,1.50], theta in [0,pi]. This includes oriented compression,
  extension and shear, with J >= .3025. Only macroscopic admissibility is assured.
- 4,096 shared scrambled Sobol states (seed 20260919) in logarithmic stretches
  and orientation; 64 unit b directions, minimizing over a by acoustic eigenvalue.
- Eight lowest cloud minima per seed initialize bounded Powell minimization
  over the two logarithmic stretches, orientation and b angle: max 2,000
  evaluations each, xtol=1e-6, ftol=1e-8. Retain every termination status.
- If negative candidates exist, seek a nearby-to-reference witness: minimize
  ||E||_F^2 subject to rank-one curvature <= -1 MPa, using SLSQP (eight starts,
  max 300 iterations). Starts include four closest negative cloud states and
  four lowest-curvature candidates. Bounds remain the same. Choose the closest
  feasible candidate and retain the most negative candidate separately.
- Verify negative witnesses by direct F-path autodifferentiation and centered
  energy differences at h=1e-3,3e-4,1e-4. Compare all 15 frozen models at the same
  states/directions. No negative sign inferred solely from optimizer status.
- At most three FOM target states: the nearest verified witness for each Free
  seed. Use existing working/check meshes, material law, residual/contact screens,
  and adaptive continuation from reference. Keep failures. For converged screened
  states, check consistent tangent against finite differences and compare mesh
  responses and the identical rank-one direction. A positive directional FOM
  curvature alone is not a proof of global or microscopic stability.
- Do not extend the domain/budget during this experiment. Results are exploratory;
  failure to reach a physical FOM solution remains inconclusive about material
  behavior. Do not modify the manuscript.

Output: `results/directed_search/`, with source/rule/checkpoint hashes recorded
before numerical evaluation. Internal interpretation: `reports/DIRECTED_SEARCH.md`.
