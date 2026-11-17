# Check of the brief for E57 (the sweep with a scale for each receiving unit), 17 November 2026

**This file is being written while the check proceeds; until the line "Check complete" appears at its end, it is a working record and not the finished check.**

The brief checked is `TASK_E57_THE_SWEEP_WITH_A_SCALE_FOR_EACH_RECEIVING_UNIT_20261117.md` in this folder, revision 1. Its SHA-256 when the check began (23:15 UTC on 17 November 2026) was `e39604ff359b97153a69a8cca63e4683cc2d59b27935e92ec4d19f99101e9474`.

## Working record (to be replaced by the finished document)

- 23:15 UTC. The brief, sections 2 and 4 of `PLAN.md`, `e42_cells.py`, `exp42_cells.py`, `exp3_state.SignLinear`, `arch26.py`, `e49_interface.py` and `exp49_interface.py` were read.
- Found so far (to be worked out):
  - `exp49_interface.py` already defines a function `magnitudes(m, layer, iface)`, which `run` calls at every scored row; the brief's new function has the same name.
  - `exp49_interface.run` asserts `MA_EXPECTED[setting]` and reads `SETTINGS[setting]`; neither has `G`. `main` asserts `setting in SETTINGS`.
  - `e42_tables.py` is on `$W9` as well as on `$W` (the brief says on `$W` only).
  - The fallback of check 1 (losses to 1e-6 after 2,048 updates) cannot be met by a crossing that differs in rounding only.
  - E49's check 2 shows that `MaskedSign` at every position gives E42's hash `837dadc1...` for `F10`; the specification's self-check 8 shows `MaskedSign` with a mask of ones equal to `SignLinear` with the scale `weight.abs().mean()`.
