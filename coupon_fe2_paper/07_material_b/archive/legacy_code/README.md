# Historical Material-B scripts

These scripts have been removed from the active root because the decisions or
figures they produced are complete and superseded. They remain byte-identical
to the versions named in historical reports and in `../relocation_manifest.json`.

They are **not** the current workflow. Do not run them to understand or train
Material B. The current entry points are:

- reader-facing figures: `../../paper_evidence/build_figures.py`;
- frozen data/preparation machinery: `../../protocol/`;
- future neural runner: not implemented yet.

Moved from the former root:

- Pilot/refinement diagnostics: `render_pilot.py`, `check_refinement.py`,
  `diagnose_compression.py`, `verify_reference.py`.
- Box/nonlinearity studies: `run_box_extension.py`,
  `explore_nonlinearity.py`, `audit_constant_tangent.py`.
- Superseded plotting: `plot_physical_response.py`,
  `plot_nonlinear_exploration.py`.

Some scripts assumed they lived in the former root. If a historical result must
be regenerated, restore its recorded layout first; do not edit the archived
copy and still claim its recorded hash. Current accepted evidence is generated
through `paper_evidence/`.
