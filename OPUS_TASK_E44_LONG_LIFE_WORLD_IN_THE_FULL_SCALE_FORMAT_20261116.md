# E44: the long-life world in the full-scale format — a new task set and learner for the E39 instrument, and its calibration (16 November 2026, revision 2)

Evidence level: none for the instrument; E2 for what the calibration measures. The proposal is the outside reviewer's (its memo, revision 16, section 25, on `claude/vibrant-cerf-sje726` at d0426a74), after job L1a. The specification, the generator file and the predictions are mine. Revision 1 (436d9677) was audited by the reviewer (`AUDIT_E44_LONG_LIFE_WORLD_BRIEF_20261116.md` on `claude/vibrant-cerf-sje726` at dc05d575), which reproduced the generator's digests with a second implementation, required nine changes and stated that after them no re-audit is needed. Revision 2 makes them and the recommendations; the last section lists them. **Released.** One worker, best the E39 worker, whose runner and scorer this reuses.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing but `smallworld/sw_run.py` and `smallworld/sw_score.py`.

## Why the small world is replaced

**Job L1a** (merged; the worker's figures, which the reviewer read and I did not recompute): 72 runs, twelve linear tasks each alone, at 64, 96 and 192 units, both rules, 16,384 updates. No run reaches the criterion of 0.8 on asked pairs. Every run fits its taught pairs (training loss 0.033 or less). The level on asked pairs goes with the number of taught pairs: 0.00 to 0.02 with 32 to 72 pairs, up to 0.36 with 162 to 264, up to 0.69 with 544 to 684. So the size rule stops, as the brief said it would; no birth was run. P39.10, P39.11 and P39.20 miss (I gave 192 units 60% to qualify); P39.18 holds.

**What that says, beside the full-scale results** (the reviewer's table, memo section 25.3): a network from random weights finds a rule only when a task offers enough taught pairs, and otherwise stores them. At full scale it does so from 1,980 or 3,960 pairs per operation in 3,000 to 17,000 updates and not from 792; the small world's largest task had 684. A network that has lived finds a new family's rule from 689 pairs in about 64 updates (E40). The small world's tasks were below what a learner without experience can learn, by construction. I built that world and did not compare its largest task with a tenth of a full-scale family; nor did either audit.

So the instrument is kept (schedule, asked and taught items in one stream, event files, scorer, the measures and the horizon rule) and the tasks and the learner are replaced: families in the full-scale format, and a learner that has been born.

## The world, as code

**`$K/lw_world_spec.py` on the default branch is the specification** (pure Python; run it to see a world). Copy it to `$M/smallworld/lw_world.py` and import it; do not rewrite it. Its rules in full, with S the world's seed:

- **Families.** A family is the four operations add, sub, max and min at one modulus, in E26's format. The 56 moduli from 41 to 100 that no experiment has used (not 73, 83, 89 or 97) are shuffled by `Random(f"lw|{S}|families")`; family g has the g-th.
- **Tasks.** Task 4g + j is operation j of family g, in the order add, sub, max, min. 224 tasks.
- **Codes.** A task's code is two bytes from 128 to 255, which no family and no birth has used. After the moduli, the same generator draws `sample(range(128 * 128), 224)`; task i takes the i-th value c as the bytes 128 + c // 128 and 128 + c % 128.
- **Items.** `code(2) a(2) symbol d_1..d_k b(2) =`, then two digits and the end symbol. Symbols `+ - > <` as in E26. k is 0 to 12 upper-case letters, as at full scale. The start and end symbols are the harness's (`EOS = 256`, `BOS = 257`).
- **Split.** A task's p² pairs, a outer and b inner, are shuffled by `Random(f"lw|{S}|split|{id}")`; the first `round(0.10 · p²)` are taught (168 to 1,000 pairs), by Python's `round`, which sends a half to the even number: 202, 302, 422, 562, 722 and 902 at the six moduli ending in 5. The rest are asked and never taught. The taught share and the lowest modulus (default 41) are arguments of the constructor, so that raising the lowest modulus is not a change to the generator.
- **Groups and visits.** A group is a family. A visit is 256 updates on one family (an argument). Each update takes 64 items, the first 48 taught, each from a task of the family drawn by `randrange(4)`, then the pair, then `randint(0, 12)`, then the letters, from `Random(f"lw|{S}|visit|{v}")`, as in the small world (lines 33, 34 and 36 of the E39 brief).
- **Schedule.** The small world's rule with `Random(f"lw|{S}|schedule")`: visit 0 is family 0; later a visit brings the next new family with chance 0.05, otherwise it returns to a met family, the absence class drawn first. **The last four families are never scheduled: the probe tasks.**
- **`needs_wrap`:** for add, a + b ≥ p; for sub, a − b < 0. **`mirror`:** 1 if the operation is add, max or min, a ≠ b and (b, a) is a taught pair.
- **Digest:** the small world's rule.
- **`in_turn`** (calibration only): visit v is family v.

**Why the new-family chance is 0.05, and what it gives** (the audit's count over 200 worlds, correcting mine, which had looked at three). There are 52 families to schedule; at 0.2 a life would have met them all by about visit 260. At 0.05 a life of 1,024 visits meets 36 to 52 of the 52 families, all of them in 99 of 200 worlds; of the first eight worlds, 1, 2, 3 and 5 meet all and 4, 6, 7 and 8 meet 47, 49, 45 and 41. A life of 256 visits meets 5 to 22 (median 13). A life of 64 visits meets a median of 4, and family 0 alone in 10 of 200 worlds, so quantities over families are read at 256 and 1,024 visits. A shorter life is the start of a longer one. The life starts slowly: the second family arrives at visit 31, 12 and 44 in worlds 1, 2 and 3 (`$K/OUTPUT_LW_WORLD_MAC.txt`).

**What it gives up, as the memo says:** the eighteen operations with coefficients, and lives of more than 52 families (208 tasks).

## The learner

A network that has been born on the first family of the project (the four operations mod 89, from half of its pairs), and then lives in the world.

- **The births are E41's**, already made at every width: `exp45/e41/e41_<S|B>_h<H>_f50_<seed>.pt`, loaded into `exp41_scale.NetW(H, U)`. Seed 73129's are on the default branch; the other two seeds' arrive with their workers' branches.
- **Width.** The half width (H = 256, U = 128; 317,457 parameters) if E41's three-seed table shows it reaching the 80% threshold on accuracy R at f = 0.50 with the fewest weight-items in at least two of three seeds under backpropagation; otherwise the present width. On seed 73129 the half width is the cheapest (E41_REPORT_73129.md). **The calibration below runs at the half width**; if the three-seed table says otherwise, I decide whether it is repeated at the present width.
- **World and learner.** The 48 calibration lives, and the second lap, use seed 73129's births in every world, and every receipt records the birth file's sha256. For the lives of L2, world w uses the births of seed (73129, 93683, 116089)[(w − 1) mod 3] once they are on the default branch.
- **If E41's three-seed table gives the present width,** the calibration is repeated at the present width before L2 is rostered. If E41 has not reported for the other two seeds when L2 is rostered, the rule is read on the seeds on file and the roster note says on how many.
- **Arms:** `naive` (the backpropagation birth, living by backpropagation), `sweep` (the sweep's birth, living by the sweep) and, as in E39's revision 4, `sweep-after-naive` (the sweep's life from backpropagation's birth; the audit's recommendation, since otherwise the two arms differ in the birth and in the life). The replay arms of E39 remain in the runner and are not run here.
- **Update and optimizer:** as E39: `exp26_candidate.step`'s text on the 48 taught items; one fresh `sw_net.make_opt(m, lr, wd)` for the life, as E39's runner (`exp26_candidate.make_opt` takes no rate or decay); learning rate 0.001 and decay 0.25 unless a run says otherwise.

**Two things about the standing optimizer that this world will meet, known since E43** (review 9.145 and the addendum above): under decay 0.25 whatever the stream stops pushing on is gone within about 2,000 updates, that is 8 visits; and a fresh AdamW's first steps on something new can wipe what the network answered before. The calibration reports what is kept at a family's return for that reason, and the learning rate and decay stay arguments.

## What a born learner answers before any update (my check, no training)

`$K/lw_row0_check.py`, E41's half-width births of seed 73129 on the first three families of worlds 1 and 2 (`$K/OUTPUT_LW_ROW0_MAC.txt`): on asked reduction pairs of add and sub, 0.0 to 5.2%; on asked pairs without reduction, 1 to 53%; on max and min, 2 to 49%.

So a level read on all asked items would start well above zero for a learner that has learned nothing of the new family, as E40's audit found for lived networks. **The gate and the compared quantities are read on reduction pairs**, where the start is at the floor.

## Changes to the runner and scorer (the instrument otherwise as E39's)

1. `sw_run.py` takes `--world lw` (the new world file), `--learner <file>` with `--H` and `--U` (the learner is loaded, not initialised; the name `birth` stays E39's, for a birth inside the world), and `--in-turn`.
2. **The answer given is recorded.** The event file gains three columns `g1`, `g2`, `g3` (uint16): the bytes generated for the asked item in order, 65535 where generation had ended. They are written in life, branch and solo files. `wrap` is the new world's `needs_wrap`, `mirror` its `mirror`.
3. `sw_score.py` gains, for each task of add and sub, **first-visit level on reduction pairs:** the share right over the task's asked reduction items in the last quarter of its first visit: the last 16, 32 and 64 updates at visits of 64, 128 and 256 updates, the last 8 at 32 and the last 128 at 512. The family's figure is the mean of add and sub. Beside it the scorer gives the same share over the whole first visit and over its first 16 updates. (With a fixed window of 64 updates, as revision 1 had it, the level at a visit of 64 updates would be the mean over the visit and at 256 the level reached, and at 64 the upper limit could not be passed.)
4. The ledger adds the birth: its updates (40,960 at the half width), its items, and that it was made by E41.
5. **What is kept: a list, when it is answered, and decay alone beside it.**
   - **At the end of every visit of a calibration life** the runner answers a fixed list without learning and writes every generated answer (by a state at every visit and `ask`, or within the run). The list: for add and for sub of every family taken so far, the family just visited included, the first 100 asked pairs in the split's order that need the reduction, without letters; and the birth family's accuracy R on E41's two samples. The list's digest for families 0 to 7 is in the audit's section 7.
   - **Kept** for a family at a later visit is its share right there divided by its share right at the end of its own visit. The report tables it by visits since, with the modulus.
   - **Decay alone, no run:** beside each kept figure, the same list answered by the weights the network had at the end of that family's own visit, every parameter but the byte table multiplied by 0.99975^u, u the updates since. For the half-width births decay alone leaves 91 to 94% of the birth family's reduction score after 256 updates, 62 to 77% after 512, 11 to 23% after 1,024 and none after 2,048 (the reviewer's computation).

## Checks before the calibration

1. **The generator.** `lw_world.py` is byte-identical to `$K/lw_world_spec.py` and prints `$K/OUTPUT_LW_WORLD_MAC.txt` for worlds 1, 2 and 3: the schedule figures, the first moduli, and the two digests of each world (the first 4 visits; 8 visits in turn at 64 updates). The reviewer's second implementation, written from the rules above, gives the same. The stream digest in each of the 48 calibration receipts equals the reviewer's for its world and visit length, and the schedule of 1,024 visits and the tasks of worlds 1 to 8 have the reviewer's digests (audit, section 7).
2. **Asked is never taught; no code and no modulus has been used before.** The runner asserts on every learned item that its pair is taught and not asked; the file's own assertions run only when it is run as a script.
3. **The learner is E41's.** The loaded network's `tensors_sha256` equals the birth receipt's, and its accuracies A and R on E41's panels equal that receipt's final row.
4. **Row 0.** The runner's scores of the born networks on the first three families of worlds 1 and 2 before any update equal `$K/OUTPUT_LW_ROW0_MAC.txt` to 2 points (another host's arithmetic).
5. **The small world still runs.** With `--world sw` a 4-visit life gives the stream digest it gave in L0 and the nine columns the small world's event file had.
6. **Asking does not teach; same stream for both arms; resume** (E39's checks 3, 4 and 9, on the new world).
7. **The scorer's new quantity** equals a hand computation on a hand-made event file, and the reviewer's scorer agrees on one calibration life. The hand-made file is the audit's (section 7): the scorer returns its medians under the window of item 3.
8. **Cost:** CPU-seconds per visit at each visit length, and a forecast for a life of 1,024 visits.

## The calibration (job F1; runs after the checks, without a further audit)

Worlds 1 to 8, `naive` and `sweep`, **eight new families in turn** (`--in-turn`, 8 visits), at visits of 64, 128 and 256 updates: 48 short lives, 57,344 updates in all, under an hour at the half width. Two more sets of 24 short lives, taken from the audit's recommendations and outside the gate: `sweep-after-naive`; and `naive` with decay 0 in the life, which shows whether the first-visit level and what is kept depend on the decay at all before any life is run.

**The gate, per visit length, under `naive`.** In at least six of eight worlds, the median over the second to the eighth family taken (families 1 to 7 in the file's numbering) of the first-visit level on reduction pairs is between **0.20** and 0.90; and a life of 1,024 visits at that length is forecast within 6 CPU-hours. Use the shortest visit length that passes.

Why 0.20 and not 0.10: before any update a born network is at 0 to 5% (check 4). A fresh network that has stored a tenth of a family without its rule is at 6 to 12% after 8,192 updates (E40) and up to 17% (E41), and recall of mirrored pairs alone is worth 0.05. A floor of 0.10 could be passed by storing.

**The trend of learning in a calibration life** is the least-squares slope of the family's figure against ln n over n = 2 to 8, the first family taken being n = 1.

**One further step for each way it can fail, named now.**
- Above 0.90 at 64 updates in six worlds: the same at 32 updates a visit, and at a taught share of 0.05 with 64 updates. Use the first that passes, in that order.
- Below 0.20 at 256 updates in six worlds: the same at 512 updates a visit; its life of 1,024 visits is forecast at about 5 CPU-hours under backpropagation and 6 under the sweep.
- If no length passes and neither case holds, stop and report the three medians of each world.
- After the one further step, stop and report whatever it gives.

E40 warns that the ceiling is the likelier failure: networks with three families behind them were at 70 to 84% on reduction pairs after 64 updates of a new family. Here the learner has one family behind it at its first visit and up to eight by its last.

**One second lap, at the adopted length.** Each of the 16 gate lives is continued by `branch` from its state after 8 visits over families 0, 1, …, 7 (the runner's generator, `world|S|branch|8|k`; the reviewer's stream digests are in the audit's section 7). Reported for each family: its share right on reduction items in the return's first update and first 16 updates, the list of item 5 at the end of the return, and its share over the last quarter of the return beside that of its first visit. The gate does not read it. At most 32,768 updates.

**Reported:** per world, arm and visit length, each family's first-visit level on reduction pairs and on pairs without reduction, max and min, with the modulus; the same over the whole first visit and over its first 16 updates; the family's figure, pooled over worlds, against its number of taught pairs (264 or fewer; 265 to 688; 689 or more) and against its place in turn, with the trend fitted on the logarithm of the taught pairs beside ln n; the trend over the eight families; score by age and the blind predictor's; what is kept, by item 5, with decay alone beside it; CPU-seconds per visit.

## Not in this brief

- **The lives** (job L2: `naive` and `replay` at 64, 256 and 1,024 visits, the sweep arms, three widths) follow by a one-page roster note after the calibration fixes the visit length, under the pace rules of the memo's section 17.8.
- **A learner born inside the world** on one of its own families, in place of the mod-89 birth. The births on file make the first calibration possible today.

## What this cannot show

- **A life of more than 52 families.**
- **Whether a family of 168 to 264 taught pairs can be learned by a born network.** No result on file bears on it. If the small families are at the floor and the large ones are not, the lowest modulus is raised by its argument and the calibration repeated; that is named now and is not a change to the generator.
- **Whether the gate's level reflects learning the modulus or the code.** A new family brings four new two-byte codes and one new modulus; the first-visit level mixes learning both.
- **Anything at the present width**, unless the calibration is repeated there.
- **A result here is a claim only when repeated on E26's own lives.** The format is full scale; the life's schedule is not E26's.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P44.1 | The reviewer's implementation gives the six digests of check 1 at the first attempt | 80% |
| P44.2 | Check 4: the worker's row-0 scores agree with mine to 2 points | 85% |
| P44.3 | The gate passes at some visit length of 64, 128, 256 | 65% |
| P44.4 | The shortest passing length is 64 | 30% |
| P44.5 | At 256 updates the median first-visit level on reduction pairs is above 0.90 in six of eight worlds (the ceiling) | 20% |
| P44.6 | At 64 updates it is below 0.20 in six of eight worlds (the floor) | 35% |
| P44.10 | At 64 updates it is above 0.90 in six of eight worlds | 5% |
| P44.11 | At 256 updates it is below 0.20 in six of eight worlds | 15% |
| P44.7 | Under `naive` at 128 updates, the mean first-visit level of families 5 to 7 is above that of families 1 to 3 (file's numbering) in six of eight worlds | 60% |
| P44.8 | The sweep's median first-visit level is within 0.15 of `naive`'s at the adopted length, in six of eight worlds | 45% |
| P44.9 | At the adopted length, the median over families 0 to 6 of a family's share right on the list of item 5 one visit after its own, divided by its share at the end of its own visit, is 0.20 or less, in six of eight worlds under `naive` | 80% |

## Done

The changed runner and scorer with the world file; the check receipts; the 48 calibration lives, the two further sets of 24, the second lap (and any further step); and `E44_REPORT.md`, which lists the committed files from `git ls-tree` (the event files and receipts of all lives and of the second lap, the answers of item 5, and the states after 8 visits of world 1), with the gate, the adopted visit length, the listed figures and the outcome of each prediction. No readings beyond these.

## What changed from revision 1 (the audit)

Made as the audit wrote them. The generator file changed in two places that leave every digest as it was (I reran it): the lowest modulus became an argument, and a sentence of its header that described something it does not do was removed. A compiled cache file I had committed by mistake is removed.

| | Change | Required? |
|---|---|---|
| 1 | The level is read over the last quarter of a first visit, with the whole visit and the first 16 updates beside it. My fixed window of 64 updates meant something different at each visit length and could not pass the upper limit at 64. | yes |
| 2 | The gate's floor is 0.20: a floor of 0.10 could be passed by storing pairs without the rule. The families it counts and the trend are defined; three predictions restated. | yes |
| 3 | Every way the gate can fail has a named end | yes |
| 4 | What is kept is read on a fixed list answered at the end of every visit, with decay alone beside it | yes |
| 5 | One second lap over the eight families at the adopted length | yes |
| 6 | The event file records the bytes of the answer given, after E43 | yes |
| 7 | Checks 1 and 7 against the reviewer's digests and its hand-made event file | yes |
| 8 | Statements corrected: how many families a life meets (I had looked at three worlds; over 200, a life of 1,024 visits meets all 52 in half of them); the optimizer's constructor; which files may be edited; where the assertions run; the rounding of the split | yes |
| 9 | The calibration uses seed 73129's births in every world, and what happens if E41's table gives another width or is incomplete | yes |
| — | A set without decay and a third arm, outside the gate; levels against the number of taught pairs; `--learner`; two more predictions; one more limit | recommended |
