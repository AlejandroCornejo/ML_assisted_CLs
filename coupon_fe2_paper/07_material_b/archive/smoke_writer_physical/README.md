# Completed bounded writer tests

Each folder moved intact from `07_material_b/work/<same-name>` on
15 September 2026. No state was deleted or resampled. These are smoke tests,
not the full campaign or an additional material-domain acceptance decision.
NPZ hashes checked before and after relocation:

| Folder | Role | States | labels.npz SHA-256 |
|---|---|---:|---|
| smoke_test_physical_v3 | two states, pause, resume and consolidation | 4 | 1778f1bc456f5ae364713197851a089e30ad2c8fb73a0357a1b87335a3c0bb11 |
| smoke_audit_working_v3 | working-mesh audit branch | 1 | f31759ad211839db0de3acec650b0df660a802bf6b2df6e9094ba79dbeb83e0f |
| smoke_audit_check_v3 | check-mesh audit branch | 1 | f59069e72faa6277bb706e2aa4dc329d459f09bbb2a88d4e178fa51370d2cb98 |
| smoke_cold_physical_v3 | independent zero-start branch | 1 | 8b1039d5e7e5149ad917631872c677cdf2f523504c34bb5b065698210d6bde25 |
| smoke_fit_physical_v4 | fit without unused tangents | 4 | 82a08ff17b67d0e3ff6061a8c6e9bed40f0d9d4b13c71fc805ea3705429f8095 |

All available states passed their residual, positive quadrature determinant,
periodic-jump and cavity-polygon screens. The failed v3 fit serialization
attempt is retained separately in `../smoke_writer_failures/`.
The v3 drivers precede the absent-tangent JSON correction; the full campaign
uses the corrected writer. These historical folders must not be resumed with
the current driver or reported as current-source bitwise replays.
