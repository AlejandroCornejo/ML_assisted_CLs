# Repository portability and size audit — 2026-09-08

## Pull failure

HEAD contained the root-level path `j, ww_nz \* area_nz`, introduced by
commit `76378cf5`. Its backslash and asterisk are invalid in Windows paths.
The 709-byte file was accidental search output, not a scientific input.
It is removed by the portability fix and remains recoverable in history.
This matches the reported `invalid path` error; the colleague's exact
terminal output and operating system have not yet been supplied.

The fix must reach the shared remote before colleagues can fetch it.
No history rewrite or disabling of Git's filesystem protections is needed
for checking out the corrected tip. Checking out older affected commits on
Windows can still encounter the invalid path.

## Measured local state before the fix

- Current tracked blobs: approximately 1.45 GiB across 5,957 paths.
- Packed Git objects/history: 5.97 GiB.
- Eight temporary pack files reported as garbage: 3.80 GiB.
- Total local `.git` disk usage: approximately 9.8 GiB.
- 4,518 untracked files, including local Python installations, large datasets,
  and scientific code/results. One untracked dataset alone was 1.69 GB.

These are local measurements, not the remote repository's billed storage or
the compressed size of a future upload. The temporary pack files have not
been removed, and no garbage collection or history rewriting was performed.

## Scope of the maintenance commit

The `.gitignore` additions exclude local Python environments, LaTeX build
intermediates, and explicitly identified large local datasets/checkpoints.
Data remain on disk. They need a separate, documented distribution mechanism
before another computer can reproduce workflows depending on them.
Small NPZ results/models, manuscript PDFs, and source files are not broadly
excluded. Already tracked files remain tracked even if an ignore rule matches.

Run `python3 tools/check_git_portability.py` after staging and before committing.
It checks the index for Windows-invalid path components, case collisions,
unmerged entries, and objects over 100 MiB. It is not an installed Git hook,
does not enforce a total repository size limit, and does not replace testing
the software on Windows.

Existing scientific/manuscript edits, deleted simulation inputs, and untracked
results were intentionally not swept into this maintenance commit. Publishing
that work calls for a separate reviewed commit and a data-distribution plan.
Reducing existing history size would require a separately coordinated cleanup;
adding ignore rules alone cannot shrink old commits.
