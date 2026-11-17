# Check of the brief for E59 before its release (17 November 2026)

**State of this document: in progress (started 22:22 UTC). The verdict below is provisional until this line is removed.**

Checked: `TASK_E59_A_TASK_OF_ANOTHER_KIND_FROM_A_FRESH_A_BORN_AND_A_LIVED_NETWORK_20261117.md` (revision 1, commit 1ed3d41c), against the default branch `origin/codex/minifly-consolidated-20261023` at 7cabb13a. Read-only; nothing was trained.

## Working notes (to be replaced by the final text)

- A (figures of E43): the counts at updates 64, 256 and 1,024, the rows on file and "to within 4 items" were recomputed from the twelve receipts and agree with the brief.
- B3/D (admission): `exp26_candidate.admit_e26` called directly fails on the default branch, because `exp16_integrated.py` has another hash than the one in the births' receipts (receipt f4559971..., default branch c9fcedcf...). E43 admits through `exp38_erase.admit`, which carries the one recorded exception. The brief names the wrong function in "The runner" and in check 3.
- B1: E43's `run` contains no `T.manual_seed`; E40's `run` calls `T.manual_seed(seed)` before `start`. Whether this matters depends on whether training draws from torch's generator (being checked).
