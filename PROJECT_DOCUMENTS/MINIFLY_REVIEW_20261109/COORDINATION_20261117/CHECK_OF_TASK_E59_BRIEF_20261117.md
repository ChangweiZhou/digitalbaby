# Check of the brief for E59 before its release (17 November 2026)

**State of this document: in progress (started 22:22 UTC). While this line stands, the check is not finished and the verdict is provisional.**

Checked: `TASK_E59_A_TASK_OF_ANOTHER_KIND_FROM_A_FRESH_A_BORN_AND_A_LIVED_NETWORK_20261117.md` (revision 1, commit 1ed3d41c), against the default branch `origin/codex/minifly-consolidated-20261023` at 7cabb13a. Read-only; nothing was trained.

## Working notes (to be replaced by the final text)

Provisional verdict: release after the listed changes.

Found so far (each recomputed; scripts in the scratch folder, to be appended):

1. `exp26_candidate.admit_e26` called directly FAILS on the default branch: the births' receipts hold `exp16_integrated.py` at f4559971..., the default branch has it at c9fcedcf.... E43 admits through `exp38_erase.admit`, which carries the one recorded exception. The brief names the wrong function twice (runner; check 3).
2. Taught panel: `e43_task.panel` cannot be used for n < 2,048 (it calls `split(seed)` itself), and sampling 300 of 256 raises ValueError. Construction that works for all n and is E43's panel at 2,048 (verified, three seeds): `rng = Random(f"e43|{seed}|panel|taught"); [render(a, b, 12, rng) for a, b in rng.sample(taught[:n], min(300, n))]`.
3. Chances: exact 1.56% (not 1.8%) for 256 and 0.018% (not 0.03%) for 512; the brief's figures are exp(-4) and exp(-8). Realised: at n = 256, symbols absent from the first/second place: 2/1 (73129), 1/1 (93683), 0/1 (116089); held-panel items with such a symbol: 11, 6, 2 of 300 (ceilings 289, 294, 298). At 512 none absent.
4. E40: lived 77.0 to 94.5 (brief: "77 to 95"; 94.48 rounds to 94), fresh 6.0 to 11.8. Cost 974 s is the largest of E40's twelve (851 to 974). E43's own runs: 0.100 to 0.110 CPU-s an update.
5. Rows: 89 confirmed; windows of "has flattened" are 8 rows each. "Three consecutive rows" spans 64 updates up to a first row of 960, 160 at 992, 256 from 1,024; the last row at which a crossing counts is 7,936. Fresh networks on a family task are below 10% until update 1,024 (E40, E41), so the fresh side is likely to cross in the coarse part and the born side in the fine part.
6. E42's `describe` is used with the run length reduced by two rows for the three-row criterion (`thr_desc`, lines 79-80); the brief says 8,192 on both sides; it should be 7,936. The two-of-three rule is undefined when a seed's word is "not decided" or "neither reaches"; "otherwise about the same" then mislabels unmeasured cases.
7. The byte table's rows 128 to 225 are bit-equal in fresh (read from E40's fresh runs), born S/B and lived S/B, for each seed. The other maps differ in size: l1.weight rms 0.20 fresh, 0.05 born; recurrent 0.038 against 0.059-0.071; l3.weight 0.029 against 0.061-0.080; l3.bias 0.029 against 0.24-0.42 (seed 73129).
8. "Has flattened" licenses the sentence on pairs when a fresh network is flat at chance without having learned the taught pairs. On E41's 23 fresh curves below 80 at 8,192 the rule calls 9 flattened, all with the taught panel at 96.8% or more; one rose by 5 points or more later, none by 10.
9. `.gitignore` at the root of the default branch excludes `*.pt`: final weights need `git add -f`. `refuse_existing` refuses a restart while a partial receipt exists.
10. B2 compares the sweep with back-propagation; the brief fixes no statement of the two rules against each other except P59.6. Nearest record for P59.6 (E41, fresh, half of the pairs, present width): ratio sweep/back-propagation 1.10, 1.19, 1.24.
11. Check 1 holds in principle: nothing between updates in E43's runner touches the optimizer, the model or a generator used in training; training draws no random numbers. E51's check 6 reproduced E43's row 64 under both rules in another container.
