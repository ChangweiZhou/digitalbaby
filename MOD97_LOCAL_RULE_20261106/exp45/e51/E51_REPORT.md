# E51: a new task learned in cells held in reserve, and how much rehearsal keeps the first family (seed 73129)

Evidence level E2. A trial on one seed, carried out from the brief `TASK_E51_A_NEW_TASK_LEARNED_IN_CELLS_HELD_IN_RESERVE_20261117.md` (revision 2) on branch `claude/vibrant-cerf-sje726`, written by the programme's outside reviewer acting for the coordinator. The worker's branch is `claude/e51-reserve-017dSEpW`, started from `codex/minifly-consolidated-20261023` at `7cabb13a`. No formal checker gate, official seal or external review is claimed. Nothing in this report concerns more than one seed.

All paths below are relative to `MOD97_LOCAL_RULE_20261106/`. Host: Linux x86-64, 4 cores, Python 3.13.18, torch 2.15.0+cpu, one thread a process, at most three processes. Every receipt records its host, the hashes of its sources and the git state.

## 1. Terms

- *The first family* is the four operations add, sub, max and min under modulus 89 (code bytes `p q v w`). Its *reduction pairs* are the pairs of add and sub whose answer needs the modulus (a + b ≥ 89; a − b < 0): 3,916 of each, 7,832 together. The *reduction score* is the mean of the two exact-answer percentages over all reduction pairs without letters (E43's score 1).
- *The new task* is either `unrelated` (E43's task on bytes 128 to 225: two symbols, 0 to 12 fillers, a query byte; the answer repeats the two symbols) or `conflict` (the second family, modulus 97, alone). Its score on *never-taught items* is E43's held panel (300 items) for `unrelated` and the two panels add97 and sub97 together (800 items) for `conflict`.
- *The rules.* B is back-propagation (the exact gradient); S is the sweep, which passes the error back through each linear map by the signs of its weights times one scale for the map.
- *The embedded network* (Part A) is E41's half-width birth (H = 256, U = 128, half of the pairs taught, `exp45/e41/e41_<rule>_h256_f50_73129.pt`) placed inside a network of the present size (H = 512, U = 256) by the brief's code. Cells 0 to 127 are the *old cells*, cells 128 to 255 the *reserve cells*; units 256 to 511 of `l1` and of the readout layer z2 are *added units*.
- *The start with a bias*: the reserve cells' update gates are shut by a bias of 10. *The keyed start*: the forward pass holds the reserve cells' update gates at 1 and the added units of z2 at 0 on every item whose first byte is one of the 46 *bytes in use* (the bytes of 2,080 taught items of the first family). The condition is computed from the item; it is a device of the harness, not something the network does by itself.
- *The rule* (arm `keyed`): the change of each old row of the input map, the recurrent matrix, `lm` and `l2` is projected away from the *directions in use* (every direction spanned by the inputs that the 2,080 items present to that matrix, with a constant 1 for the bias), both on the gradient and on the step; `l1`, `l3`'s old columns, `l3`'s bias and the 46 rows of the byte table in use are restored after every step.
- *The probe* (Part B): 400 reduction pairs without letters (200 add, 200 sub, drawn by `Random("e51|73129|probe")`) and E43's held panel of the unrelated task.

## 2. Code

| File | What it is |
|---|---|
| `exp51_reserve.py` | Part A's runner (new). The forward pass is `arch26.forward`'s text with the brief's two added lines; the update is `exp26_candidate.step`'s text with the rule's two additions. `exp43_unrelated.py`, `e43_addendum2.py` and `exp41_scale.py` are imported, not edited. After `arch26.install()` it sets `exp11_cell.forward` to its own copy, so the scorer's generation runs the keyed forward. |
| `exp51_partb.py` | Part B's runner (new). E43's setting: `exp38_erase.step` on `arch26.forward`, the E26 birth admitted by `exp38_erase.admit`. It is a second file so that `exp51_reserve.py`'s hash in Part A's receipts stays fixed; the reviewer accepted this. |
| `e51_read.py` | The reader: computes every reading of sections 4 and 6 from the receipts and writes `exp45/e51/E51_READ_A.json`, `E51_TABLES_A.md`, `exp45/e51/partB/E51_READ_B.json`, `E51_TABLES_B.md`. |

No existing source file was edited. Commands: `python3 exp51_reserve.py start <bias|keyed> <S|B> 73129`; `python3 exp51_reserve.py check <1..6> 73129`; `python3 exp51_reserve.py run <task> <arm> <S|B> 73129`; `python3 exp51_partb.py <share|intervals|steps|check> ...`; `python3 e51_read.py <a|b> 73129`.

## 3. Part A: the checks before the runs

All seven passed for both rules. Receipts in `exp45/e51/checks/`.

1. **The embedding** (`e51_check_1_embedding_73129.json`). The keyed start's outputs at the answer positions equal the half-width birth's with a largest difference of exactly 0.0, on the 1,200 held-out items at 12 letters and on the 2,000 sampled items without letters, under both rules (limit 1e-5). The start with a bias differs by at most 0.0029 and 0.0010 (S) and 0.0034 and 0.0011 (B) (limit 0.02). The scores of row 0 equal the half-width birth's on every panel.
2. **The directions** (`e51_check_2_directions_73129.json`). The digest of the 2,080 items begins `d22b5733778491b1`, as the reviewer's; 46 bytes in use; 46, 129, 129 and 11 directions at the input map, the recurrent matrix, `lm` and `l2`, under both rules.
3. **The rule** (`e51_check_3_rule_73129.json`). After 8 updates of `keyed` the largest change of an output on the panels of check 1 is 1.7e-5 (S, unrelated), 1.9e-5 (S, conflict), 1.7e-5 (B, unrelated) and 2.1e-5 (B, conflict), all below 1e-4; `l1`, `l3`'s old columns, `l3`'s bias and the 46 rows of the byte table in use are bit-equal to the start. With the restoring of the byte table switched off, 41 of the 46 rows move on the conflict stream (outputs then change by up to 0.31). The same 8 updates of `keyed-free` change outputs by 20 to 39. On a batch of the first family every reserve state and every added unit of z2 is exactly 0; on batches of either new task they are not (largest 0.73 to 0.93). The changes of 1e-5 under the rule are float32 rounding of the projected step, not a leak: the generated answers of the first family stay equal to row 0's throughout every `keyed` run (section 4).
4. **The scorer** (`e51_check_4_scorer_73129.json`). In the keyed start, 0 of the 32,884 generated answers of the first family (all pairs without letters and the held-out panel) differ from the half-width birth's. With the two added lines switched off, 1,372 (S) and 767 (B) differ, so the lines act in generation.
5. **The data** (`e51_check_5_data_73129.json`). The batch digests at update 4,096 equal E43's `unrelated` receipt (`f6011054...`), E43's `conflict` receipt (`60870058...`) and the second addendum's `unrelated-rehearse` (`11fcebdb...`), under both rules.
6. **`open` is E43 in another network** (`e51_check_6_open_is_e43_73129.json`). Started from E26's birth as it is, the runner of `open` reproduces E43's `unrelated` receipt at row 64: the first family's counts, the new task's counts, the data digest and the tensors hash (`7c98af37...` for S, `60c882bd...` for B).
7. **Cost** (`e51_check_7_cost.json`, from sixteen dry runs of 64 updates in `exp45/e51/dry/`): 2.76 CPU-hours forecast. The sixteen runs took 2.74 CPU-hours.

## 4. Part A: results

Every run at every scored row is in `exp45/e51/E51_TABLES_A.md`; the readings are in `E51_READ_A.json`. The brief's readings: *kept* means the reduction score at update 4,096 is within 2 points of row 0; *learned* means the never-taught score reaches 80% at a scored row and stays there. The brief's third reading (called there "learned in the reserve") asks that a learned task's never-taught score at update 4,096 with the reserve cells held shut on every item be below 20%; at the reviewer's request (section 4.1) that name is not used here, and the column gives the criterion and the figures. The reader's files `E51_TABLES_A.md` and `E51_READ_A.json`, written before that request, keep the brief's name for the column and the key (`learned_in_the_reserve`); they record the same criterion.

| task | rule | arm | reduction row 0 | at 64 | at 4,096 | kept | new at 4,096 (%) | first row learned | new, reserve shut (%) | reserve-shut score below 20% |
|---|---|---|---|---|---|---|---|---|---|---|
| unrelated | S | `open` | 92.4 | 0.0 | 0.0 | no | 100.0 | 512 | 0.7 | yes |
| unrelated | S | `rehearse` | 92.4 | 35.3 | 90.0 | no | 97.3 | 512 | 0.0 | yes |
| unrelated | S | `keyed-free` | 92.4 | 0.0 | 0.0 | no | 100.0 | 512 | 0.0 | yes |
| unrelated | S | `keyed` | 92.4 | 92.4 | 92.4 | yes | 100.0 | 1,024 | 0.0 | yes |
| unrelated | B | `open` | 95.3 | 0.0 | 0.0 | no | 100.0 | 512 | 5.7 | yes |
| unrelated | B | `rehearse` | 95.3 | 72.8 | 92.8 | no | 100.0 | 512 | 0.0 | yes |
| unrelated | B | `keyed-free` | 95.3 | 0.0 | 0.0 | no | 100.0 | 512 | 0.3 | yes |
| unrelated | B | `keyed` | 95.3 | 95.3 | 95.3 | yes | 100.0 | 512 | 0.0 | yes |
| conflict | S | `open` | 92.4 | 12.4 | 1.3 | no | 85.0 | 1,024 | 83.0 | no |
| conflict | S | `rehearse` | 92.4 | 54.8 | 88.3 | no | 85.9 | 2,048 | 83.0 | no |
| conflict | S | `keyed-free` | 92.4 | 16.1 | 1.0 | no | 89.6 | 2,048 | 83.6 | no |
| conflict | S | `keyed` | 92.4 | 92.4 | 92.4 | yes | 94.8 | 1,024 | 41.0 | no |
| conflict | B | `open` | 95.3 | 12.7 | 0.0 | no | 94.9 | 1,024 | 87.6 | no |
| conflict | B | `rehearse` | 95.3 | 65.5 | 97.8 | no | 90.9 | 2,048 | 66.6 | no |
| conflict | B | `keyed-free` | 95.3 | 17.4 | 0.2 | no | 92.6 | 1,024 | 78.1 | no |
| conflict | B | `keyed` | 95.3 | 95.3 | 95.3 | yes | 96.9 | 512 | 40.4 | no |

**`keyed` keeps the first family's answers.** In all four `keyed` runs every one of the 31,684 generated answers of the first family without letters, and every answer of the held-out panel at 12 letters, equals row 0's at every scored row up to update 4,096 (share of answers equal to row 0: 100.00%). The reduction score therefore stays at 92.4 (S) and 95.3 (B). No other arm keeps it by the 2-point criterion. The answers are unchanged; the outputs are not exactly unchanged (section 4.1, statement 1).

**`keyed` learns both new tasks.** On `unrelated` it reaches 100% on the never-taught items under both rules (S: 43.0% at update 512, 100% from 1,024; B: 91.3% at 512). On `conflict` it reaches 94.8% (S) and 96.9% (B) at update 4,096, the highest of the four arms under each rule.

**The cost of the rule and of the device.** The ratio of first rows learned:

| task | rule | `keyed` | `keyed-free` | `open` | cost of the rule (`keyed` / `keyed-free`) | cost of the device (`keyed` / `open`) |
|---|---|---|---|---|---|---|
| unrelated | S | 1,024 | 512 | 512 | 2 | 2 |
| unrelated | B | 512 | 512 | 512 | 1 | 1 |
| conflict | S | 1,024 | 2,048 | 1,024 | 0.5 | 1 |
| conflict | B | 512 | 1,024 | 1,024 | 0.5 | 0.5 |

The scored rows are spaced by factors of two, so a ratio of 2 or 0.5 means one row later or earlier.

**Which connections the new task needs.** On `unrelated`, with the reserve cells held shut, the never-taught score at update 4,096 is 0.0% in `keyed` and 0.0 to 0.3% in `keyed-free`. On `conflict`, with the reserve cells held shut, it is 41.0% (S) and 40.4% (B) in `keyed`, and 78 to 88% in `keyed-free` and `open`. The reviewer's further figures (section 4.1, statement 3) show that the unrelated task also needs the connections from the reserve cells into the old units, and that on `conflict` the part that survives shutting the reserve is the items whose answer is the same under both moduli.

**The reserve's figures at update 4,096** (`E51_READ_A.json`, `reserve_4096`). Reserve cells whose update gate is open by more than 0.1 on the new task's panel: `keyed` 119 (S) and 125 (B) on `unrelated`, 112 and 83 on `conflict`; `keyed-free` 125, 122, 118 and 37. In `open` and `rehearse`, which start with the gates shut by a bias, 0 to 8 cells are open by more than 0.1; yet in `open` on `unrelated` shutting the reserve brings the new task from 100% to 0.7% (S) and 5.7% (B), so the few slightly opened reserve cells (mean opening 0.047 and 0.061) are what carry it. In the keyed arms the largest reserve state on the first family's panel is exactly 0; in `open` and `rehearse` it is 0.71 to 1.0, that is, the reserve is in use on the first family's items there.

**The wrong answers on reduction pairs at update 4,096** (classes of the brief, add and sub together, 7,832 pairs):

| task | rule | arm | end symbol alone | byte ≥ 128 | right under 97 | off by 1 or 10 | other two digits | other |
|---|---|---|---|---|---|---|---|---|
| unrelated | S | `open` | 7,287 | 545 | 0 | 0 | 0 | 0 |
| unrelated | S | `keyed-free` | 7,832 | 0 | 0 | 0 | 0 | 0 |
| unrelated | B | `open` | 6,929 | 903 | 0 | 0 | 0 | 0 |
| unrelated | B | `keyed-free` | 6,979 | 824 | 0 | 0 | 0 | 29 |
| conflict | S | `open` | 0 | 0 | 4,510 | 239 | 2,980 | 0 |
| conflict | S | `keyed-free` | 0 | 0 | 3,194 | 359 | 4,137 | 64 |
| conflict | B | `open` | 0 | 0 | 5,393 | 360 | 2,070 | 9 |
| conflict | B | `keyed-free` | 0 | 0 | 4,403 | 799 | 2,612 | 0 |

After the unrelated task the network answers a first-family prompt with the end symbol, or with a byte of the unrelated task; after the second family it answers with two digits, most often the answer under modulus 97. The `keyed` runs' wrong answers are those of row 0 (463 and 131 in classes 4 and 5 under S; 347 and 25 under B). The full table, with `rehearse`, is in `E51_TABLES_A.md`.

### 4.1 Three statements on Part A, from the reviewer's verification

The reviewer verified the trial from these receipts and weights (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/E51_TRIAL_VERIFICATION_20261117.md` on branch `claude/vibrant-cerf-sje726`, commit `e6333d0b`) and asked that three statements stand in this report. The figures in them are the reviewer's measurements, not this worker's.

1. **The first family's answers are unchanged; its outputs are not.** By row 4,096 the first family's outputs have moved by up to 5.3e-4 (3.8e-4 and 5.3e-4 in the two `unrelated` runs, 5.5e-5 and 7.3e-5 in the two `conflict` runs). The weights are stored in single precision after each projected step, so a small part of the change remains in the directions the first family uses and grows with the number of updates. The smallest difference between the largest and the second largest output of a first-family answer is 1.8e-4 under the sweep (7.8e-4 under back-propagation). No answer changed in 4,096 updates; nothing in the construction bounds this movement over a longer run.
2. **The separation is made by the program.** The two added lines of the forward pass read each item's first byte and, when it is one of the 46 bytes in use, shut the reserve cells' gates and zero the added units of the readout. With the two lines removed, the same weights of row 4,096 answer 0.00% of the reduction pairs after `unrelated` and 0.40% and 0.00% after `conflict`. The first family is kept by the weights together with that program, not by the weights alone.
3. **The new task is not learned in the reserve alone.** For `unrelated` the score with the reserve cells shut is 0%, but it is also 0.0% and 0.3% when the incoming connections of the old cells and old readout units are put back to their starting values; the task needs the connections from the reserve cells into the old units. For `conflict` the never-taught score with the reserve cells shut is 41.0% (S) and 40.4% (B); by the reviewer's count, the items answered then are those whose operands are below 89 and whose answer is the same under both moduli.

### The predictions of Part A

| | Statement | Chance | Outcome |
|---|---|---|---|
| P51.1 | `open` loses the first family on `unrelated`: below 5% by update 64, both rules | 90% | Held (0.0 and 0.0) |
| P51.2 | `open` loses the reduction pairs on `conflict` (below half of row 0 at 4,096), both rules | 80% | Held (1.3 and 0.0) |
| P51.3 | `keyed` keeps the first family on both tasks, both rules | 97% | Held (every answer equal to row 0's) |
| P51.4 | `keyed` learns `unrelated` | 60% B, 45% S | Held under both rules |
| P51.5 | `keyed` learns `conflict` | 50% B, 35% S | Held under both rules |
| P51.6 | Where `keyed` learns `unrelated`, the cost of the rule is at most four | 60% | Held (2 under S, 1 under B) |
| P51.7 | `rehearse` keeps the first family within 5 points, both tasks, both rules | 85% | Held (2.4, 2.5, 4.1 below row 0; 2.5 above) |
| P51.8 | `keyed-free` does not keep the first family on `unrelated`, either rule | 85% | Held (0.0 and 0.0) |
| P51.9 | Where `keyed` learns `unrelated`, its score with the reserve shut is below 20% (the brief's "learned in the reserve") | 75% | The criterion is met (0.0%, both rules); see section 4.1, statement 3, for what it does not show |

## 5. Part B: the checks before the runs

Receipts in `exp45/e51/partB/checks/`; all passed.

8. **B1's runner is the second addendum's at k = 2** (`e51_check_8_b1_is_addendum_73129.json`): with the addendum's own choice the runner gives the addendum's first-family counts at row 64 and its tensors hash at update 4,096 (S `15dc8293...`, B `f89d3089...`), and the same data digest. The two runs are in `partB/check8/`.
9. **B3 at 0.001 is E43's run** (`e51_check_9_b3_is_e43_73129.json`): the first-family counts at rows 64 and 4,096 and the tensors hash at 4,096 equal E43's `unrelated` receipt (S `88b52850...`, B `7c70f8b0...`); the rate read back from both groups is 0.001.
10. **The probe** (`e51_check_10_probe_73129.json`): at update 0 the probe's 400 items are answered correctly 381 times (S) and 393 times (B), the same counts as those items in score 1's own generation.
11. **Cost** (`e51_check_11_cost_73129.json`): 6.1 CPU-hours forecast (B2 at its upper limit); Part B took 4.95 CPU-hours.

The order of events: check 10 was run first; the queue of Part B then began with the runs of checks 8 and 9 and the two runs of B3 at 0.0001, and the receipts of checks 8, 9 and 11 were written as soon as those runs had finished, by which time the first run of B1 had begun.

## 6. Part B: results

The tables are in `exp45/e51/partB/E51_TABLES_B.md`, the readings in `E51_READ_B.json`.

### B1. The share of rehearsal

| rule | share | reduction row 0 | at 4,096 | % of row 0 | kept (within 5 points) | lowest probe (%) | unrelated held at 80%: first probe row | unrelated held at 4,096 (%) |
|---|---|---|---|---|---|---|---|---|
| S | 1/2 (second addendum) | 96.6 | 97.1 | 100.5 | yes | – | – | 99.3 |
| S | 1/4 | 96.6 | 89.0 | 92.2 | no | 55.5 | 512 | 98.7 |
| S | 1/8 | 96.6 | 86.8 | 89.8 | no | 30.5 | 512 | 99.0 |
| S | 1/16 | 96.6 | 63.2 | 65.4 | no | 19.8 | 448 | 100.0 |
| S | 1/32 | 96.6 | 44.2 | 45.8 | no | 7.0 | 448 | 100.0 |
| S | 0 (E43 `unrelated`) | 96.6 | 0.0 | 0.0 | no | – | – | 100.0 |
| B | 1/2 (second addendum) | 98.1 | 96.7 | 98.6 | yes | – | – | 99.3 |
| B | 1/4 | 98.1 | 88.9 | 90.6 | no | 61.5 | 256 | 100.0 |
| B | 1/8 | 98.1 | 78.4 | 79.9 | no | 40.5 | 256 | 99.7 |
| B | 1/16 | 98.1 | 57.5 | 58.6 | no | 25.8 | 256 | 100.0 |
| B | 1/32 | 98.1 | 34.6 | 35.3 | no | 18.0 | 256 | 100.0 |
| B | 0 (E43 `unrelated`) | 98.1 | 0.0 | 0.0 | no | – | – | 100.0 |

**The least share that keeps is 1/2 under both rules.** A share of 1/4 leaves the reduction score 7.6 (S) and 9.2 (B) points below row 0. What is left falls steadily as the share falls; every share leaves far more than none. The unrelated task is learned at every share, as fast as or faster than with a share of 1/2 (the second addendum's rows are 0, 64, 256, 1,024, so its first row at 80% is not on the probe's grid and is not given).

### B2. Rehearsal at intervals

One optimizer for the whole run; eight cycles of a gap of 256 unrelated updates and a review on the first family's taught stream until the probe is at 90% of its value at update 0 (S 95.25%, goal 85.73%; B 98.25%, goal 88.43%). Every review reached its goal within the limit of 1,024.

| cycle | S: probe at end of gap (%) | S: review updates | B: probe at end of gap (%) | B: review updates |
|---|---|---|---|---|
| 1 | 0.00 | 80 | 0.00 | 80 |
| 2 | 0.00 | 128 | 0.25 | 64 |
| 3 | 3.75 | 96 | 11.75 | 64 |
| 4 | 52.50 | 96 | 46.75 | 64 |
| 5 | 74.50 | 64 | 70.00 | 32 |
| 6 | 65.00 | 48 | 46.50 | 48 |
| 7 | 54.25 | 48 | 77.75 | 48 |
| 8 | 58.25 | 32 | 72.50 | 48 |

The unrelated task's held score at the end of each gap is 98 to 100% from cycle 2 on (S, cycle 1: 6.0%; B, cycle 1: 98.0%). The reviews cost it as well: at the end of the reviews of cycles 1 to 3 it is 0 to 35% (S) and 0.3 to 70% (B), and from cycle 4 on 88 to 100%. At the end of the run (4,616 updates under S, 4,448 under B) the reduction score is 87.9 (S) and 91.9 (B) and the unrelated held score 299/300 and 294/300. Each cycle's probes are in the receipts `e51_B2_intervals_<rule>_73129.json`.

- **S.** The review shortens (cycle 8: 32 updates, at most half of cycle 1's 80). More is left (58.25 against 0.00 points at the end of the gap). *With rehearsal at intervals the first family becomes cheaper to restore, and more of it survives a gap.*
- **B.** The review does not shorten (48 updates against 80; half is 40). More is left (72.50 against 0.00). *With rehearsal at intervals the first family does not become cheaper to restore, and more of it survives a gap.*

The explanation under test ("in the present network a memory does not become more lasting by being rehearsed; each gap removes it as the first did") is refuted under both rules by its own criterion. What is left after a gap is not monotone in the cycle (S: 74.5, 65.0, 54.3, 58.3 in cycles 5 to 8), and this is one seed.

### B3. Smaller steps on the new task

| rule | rate | updates | first probe row with the unrelated task at 80% | first-family probe there (%) | last probe row with the first family at half of update 0 or more | unrelated held there (%) | reduction at the end | unrelated held at the end (%) |
|---|---|---|---|---|---|---|---|---|
| S | 0.001 | 4,096 | 384 | 0.00 | 0 | 0.0 | 0.0 | 100.0 |
| S | 0.0003 | 13,824 | 960 | 0.00 | 16 | 0.0 | 0.0 | 100.0 |
| S | 0.0001 | 40,960 | 2,304 | 0.00 | 96 | 0.0 | 0.0 | 100.0 |
| B | 0.001 | 4,096 | 224 | 0.00 | 0 | 0.0 | 0.0 | 100.0 |
| B | 0.0003 | 13,824 | 448 | 0.25 | 16 | 0.0 | 0.0 | 100.0 |
| B | 0.0001 | 40,960 | 1,024 | 0.25 | 48 | 0.0 | 0.0 | 100.0 |

**Smaller steps do not leave more**, under either rule. With smaller steps the first family is lost over more updates (the probe at update 64 under S: 0.0% at 0.001, 21.5% at 0.0003, 62.0% at 0.0001), but the new task is also learned over more updates, and by the time it is at 80% the first family's probe is at 0 to 0.25% at every rate.

### The predictions of Part B

| | Statement | Chance | Outcome |
|---|---|---|---|
| PB.1 | A share of a quarter keeps the first family, both rules | 70% | Failed (7.6 and 9.2 points below row 0) |
| PB.2 | A share of 1/32 does not keep it, either rule | 65% | Held |
| PB.3 | The review shortens, under back-propagation | 60% | Failed (80 to 48 updates; half would be 40). It shortens under the sweep |
| PB.4 | More is left after a gap at cycle 8, under back-propagation | 30% | Held (72.5 against 0.0); also under the sweep |
| PB.5 | Smaller steps do not leave more, either rule | 65% | Held |

## 7. Deviations, restarts and what was not done

- **Branch.** No branch name was given; the work is on `claude/e51-reserve-017dSEpW`, accepted by the reviewer.
- **Two files.** Part B's runner is `exp51_partb.py` rather than a part of `exp51_reserve.py` (accepted by the reviewer). `e51_read.py` is a third new file, a reader of receipts only.
- **Restart.** The machine restarted while the two runs of B3 at 0.0003 were in progress. Their partial receipts were moved to `exp45/e51/partB/aside/` and both runs were started again from the beginning; the finished receipts are those of the second start. No other run of either part was restarted.
- **Check 3, the variant without restoring the byte table, on the unrelated stream.** The brief asks that at least one of the 46 rows differ on the conflict stream; on the unrelated stream no byte in use occurs but the start symbol, and no row moved. The receipt records both; the requirement is applied to the conflict stream only.
- **Weights committed.** Part A: the four starts with their directions, and the weights at updates 64 and 4,096 of every run. Part B: the weights at the end of every run (the brief names no rows for Part B). The weights of the scored rows in between are not kept. The dry runs of check 7 and the runs of check 6 have receipts and no committed weights.
- **The commit trailers** carry `Acting-Agent: claude:<session>` and the session link, as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` asks.
- **Not shown** (as the brief states): that a learner can do without the keyed device; how a reserve is shared among several tasks; any statement about the seeds 93683 and 116089; whether a network can be born with a reserve by itself; in Part B, more than two tasks or a share of rehearsal that changes with time.

## 8. Committed files

The list is produced by `git ls-tree -r --name-only HEAD MOD97_LOCAL_RULE_20261106/exp45/e51 MOD97_LOCAL_RULE_20261106/exp51_reserve.py MOD97_LOCAL_RULE_20261106/exp51_partb.py MOD97_LOCAL_RULE_20261106/e51_read.py` at the commit that adds this report, and is in `E51_COMMITTED_FILES.txt` beside it.
