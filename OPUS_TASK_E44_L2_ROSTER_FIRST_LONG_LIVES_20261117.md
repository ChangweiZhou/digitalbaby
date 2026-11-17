# E44, job L2: the first long lives — roster note (17 November 2026, revision 2, entered by the worker from the reviewer's audit at ef8b512b, in the coordinator's absence)

**What revision 2 is.** Revision 1 (entered at 30628a4c) was audited by the outside reviewer (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/AUDIT_E44_L2_ROSTER_NOTE_20261117.md` on `claude/vibrant-cerf-sje726`, commit ef8b512b), which required nine changes and stated that after them no second audit of the note is needed. The coordinator, which normally enters an audit's changes, cannot be reached; the reviewer's instructions to the worker (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/WORKER_E44_L2_FIRST_LONG_LIVES.md` on the same branch) hand the worker the note and the audit together and take the decisions the audit leaves to the coordinator. This revision enters the nine changes as the audit's section 8 words them, and those decisions; the last section lists every change. The text that is not listed there is revision 1's. "I" and "mine" in revision 1's text are the coordinator's. The coordinator has not seen this revision.

Evidence level E2. Under the pace rules of the reviewer's memo (section 17.8) a roster inside the audited protocol needs a one-page note with predictions on file and no audit; **any change to the runner is audited.** This roster needs two small runner changes (below). In revision 1 the note was released for everything except the change to the runner, which waited for the reviewer's check. The audit found the change sound as described, under four conditions that revision 2 enters; the lives start when the changed runner's checks pass against the figures of the audit's section 7, and the reviewer reads the changed code and the check receipts when they are committed (audit, section 10). Written at the user's word of 17 November.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit.

## The question

The adopted question (review 9.148): under fixed memory and continued learning, can a learner keep what it has learned while learning new things, at no greater computation to threshold than backpropagation? L2's part of it, before any candidate: **does the conventional learner get stuck over a long life, and what do the two conventional protections, a plain replay store and no weight decay, change?** The two rules are a secondary factor and come in the second release.

## What the record says (working rule 1)

- **The instrument is calibrated** (E44): visits of 256 updates; median first-visit level on reduction pairs 0.30 to 0.64 under `naive`; a life of 1,024 visits forecast at 3.85 CPU-hours. Nothing is kept from one visit to the next: one visit after a family's own, 0.7 to 7.6% of its level is left, where decay alone over those 256 updates would leave 89%. Returns end higher than first visits (0.58 against 0.49 over the last quarter).
- **Rehearsal keeps a family at no cost** under unrelated learning, with and without decay, under both rules (E43); and in a life of contradicting families (E38).
- **Learning the first family does not need decay** under either rule (E46, three seeds).
- **Decay alone erases what is not used** within about 2,000 updates (review 9.145). In worlds 1 to 8, about half of all returns in a life of 1,024 visits come after more than 8 visits (2,048 updates): 440 to 497 of 972 to 983. So decay can matter here even though it did not within one visit.
- **The width rule of the E44 brief gives the half width:** at taught fraction 0.50 under backpropagation the half width reaches 80% on R with the fewest weight-items on all three seeds (E41: seed 73129's report; seeds 93683 and 116089, 256 to 512, *rises* in weight-items).
- **A life of 1,024 visits meets 41 to 52 families** in worlds 1 to 8; one of 256 visits meets 9 to 17.
- **Under `replay` a visit teaches 6,144 new items, the number the calibration taught at 128 updates a visit, where four of eight worlds were below the gate's floor.** (Audit, change 8.)

## Learners and worlds

E41's half-width births at f = 0.50 (H = 256, U = 128): world w uses seed (73129, 93683, 116089)[(w − 1) mod 3], so worlds 1, 4, 7 use 73129; 2, 5, 8 use 93683; 3, 6 use 116089. Backpropagation's birth for the arms below. The births of the two later seeds are merged to the default branch with this note.

## Arms (release 1): what protects old knowledge, under backpropagation

| Arm | Replay store | Decay in the life | Stored numbers beyond the weights and optimizer |
|---|---|---|---|
| `naive` | none | 0.25 | none |
| `replay` | ρ = 0.5 (24 of 48 slots from the buffer) | 0.25 | 4 × 317,457 bytes: 55,209 items of 23 bytes |
| `naive-wd0` | none | 0 | none |
| `replay-wd0` | ρ = 0.5 | 0 | as `replay` |

The buffer is E39's (reservoir over every new taught item, capacity `(4 × P) // 23` items, replay from the first update at which it holds 24 items). ρ = 0.5 follows E38's and E43's rehearsal share; it is not tuned, and that is a limit. Every arm consumes 64 stream items an update.

## Runs (release 1), in this order

The order is the audit's recommendation 1: if the session is cut short, the main question keeps its lives.

| | Worlds | Visits | Arms | Lives | CPU-hours (forecast) |
|---|---|---|---|---|---|
| 0. The other two births | 2, 3 | 8, in turn | `naive` | 2 | 0.07 |
| 1. Trial | 1 | 1,024 | `naive`, `replay` | 2 | 8.4 |
| 2. Long | 2, 3, 4 | 1,024 | `naive`, `replay` | 6 | 25.2 |
| 3. Short | 5 to 8 | 256 | `naive`, `replay` | 8 | 8.4 |
| 4. Decay | 1 to 8 | 256 | `naive-wd0`, `replay-wd0` | 16 | 16.8 |
| 5. Learning probe | 1 to 4 | 4, from three ages | `naive`, `replay` | 24 branches | 0.4 |

About 59 CPU-hours: above the memo's cap of 48 by the two decay arms, which the audit of the E44 brief recommended for this roster and E46 makes cheap to read. A life of V visits is the first V visits of a longer life, so worlds 1 to 4's lives at 1,024 give their 256-visit figures, and every arm has eight worlds at 256 visits. Forecasts are from E44's measured 0.053 CPU-seconds an update under `naive`, with 10% added for the buffer, which is unmeasured on this world; check 7 measures it. The list adds about 1.2 CPU-hours over the 32 lives (audit, section 4.8).

**The other two births (audit, recommendation 3).** Before the lives, one 8-visit life in turn at 256 updates under `naive` in world 2 with the birth of seed 93683 and in world 3 with the birth of seed 116089. The calibration's medians with seed 73129's birth were 0.566 and 0.301 there. If either median first-visit level is below 0.20, the worker stops and reports: the gate would not hold for that birth.

**The trial (working rule 2).** World 1's two lives run first; a third process may start world 2's long life beside them. The worker reports their curves and diagnostics (below) and **continues without waiting** unless a stop condition holds: a loss that is not finite; the buffer below capacity after visit 64; the score by age on reduction pairs below 0.05 over visits 512 to 1,023 in both arms; or a forecast for release 1 above 75 CPU-hours. The trial's report gives the note's diagnostics, the quantities D2 to D10 and D14 of the section "What is recorded and computed", and **the median first-visit level (D6) of `replay` over the families met, beside `naive`'s. If it is below 0.20, the trend of learning is not read for `replay`, and release 2 adds `replay` at ρ = 0.25** (audit, change 8). I read the trial while the rest runs.

**Restarts (audit, recommendation 2).** The long lives run with `--save-states 64,128,256,384,512,640,768,896,1024`. An interrupted life is continued with `life --resume <latest state>`; older states may be deleted except those of the learning probe. The report says which lives were resumed.

**The lives without decay record the sizes of the weights** (the norm of each tensor) at the end of every visit at which the whole list is answered, that is at every eighth visit. E46 found the recurrent matrix still growing at the end of a birth without decay. (The reviewer's decision, taken in the coordinator's absence.) The runner is not changed for this: the lives without decay save a state at every eighth visit, and the norms are read from the states.

## The runner changes (two; audited as described; the reviewer reads the code and the check receipts when they are committed)

E44's runner answers the fixed list after **every** visit for every family taken so far, and on each family's first visit computes decay alone for **every** later visit of the life. In an 8-visit calibration life that is cheap; in a life of 1,024 visits the present rule answers 15 to 17 million items, 28 to 33% of the life's computation: 5 to 6 million for decay alone of the families, 4 million for the birth family, 5 to 6 million for the families answered by the network. Two arguments, with defaults that keep E44's behaviour:

- `--list-every K`: every family taken so far is answered at the end of every K-th visit and of the last visit; the family just visited is answered at the end of every visit. Default 1. **L2 uses 8.** "Every K-th visit" is read as the audit's section 7 reads it: the visits after which K, 2K, … visits are done.
- `--decay-distances d1,d2,…`: decay alone is computed at these distances in visits from a family's first visit only. Default: every later visit. `--decay-distances` also takes `none`. **L2 uses `--list-every 8 --decay-distances none`**: decay alone cannot be read in a life with returns (audit, 1.3 c), it leaves nothing beyond eight visits, and the two decay-0 arms carry the question. (Revision 1 used the distances 1, 2, 4, 8, 16, 32, 64, 128, 256, 512.) If decay alone is wanted nevertheless, the readable form is: from the weights at the end of each visit of a family, at the next K-th visit, and only if the family is not visited again before it; that form would need its own check.
- **The birth family.** The birth family's 2,000 items follow both arguments: the network answers them before visit 0 and at the end of every K-th visit and of the last; decay alone of the born weights is computed at the given distances, counted in visits from the start of the life.
- **The second change.** `load_learner` also accepts the replay arms: `replay` loads backpropagation's birth and `sweep+replay` the sweep's (the line `want = {…}[arm]`). Nothing else in the replay path changes. Check 3 (ρ = 0 equals `naive`) exercises it.

**Checks of the change** (the audit's change 4).

- (a) With the defaults, world 1's 8-visit calibration life at 256 updates gives, under the changed runner and under the runner at 39e4e594 on this host, the same hashes: the list's arrays, the events' arrays, the final weights and optimizer state, and the learned batches. Where this host reproduces E44's committed receipt, they are the committed hashes.
- (b) World 5, 32 visits by the schedule under `naive`, with the birth of its world (seed 93683), three times: with the defaults; with `--list-every 4 --decay-distances 1,2,4,8`; with `--list-every 4 --decay-distances none`. The row keys of each list file have the reviewer's digest (audit, section 7), and every row present in two of the runs is equal in them item by item, generated bytes included. About 25 CPU-minutes in all.
- (c) The event file and the hashes of the weights and of the optimizer are the same with and without the change (the list does not learn), on the three runs of (b).
- (d) The two arguments are entries of the life's settings; the receipt and every state carry them, and a resumed life keeps them. Check 5 compares the list file's arrays hash as well.

## Checks before the lives

1. The E44 runner at the commit used still gives check 1's digests (worlds 1 to 8, schedules and tasks).
2. The births of seeds 93683 and 116089: file hashes equal their E41 receipts, and A and R through the runner equal the receipts' final rows (as E44's check 3).
3. `replay` with ρ = 0 equals `naive` on world 1 for one visit: same weights hash and event file.
4. The buffer's capacity printed at the start is 55,209 items, and it is never exceeded.
5. Resume: a `replay-wd0` life of world 1, stopped at visit 4 of 8 and resumed, gives the uninterrupted hashes and event file, and the list file's arrays hash.
6. The decay arms read back `weight_decay` 0.0 in the optimizer's first group at the start and the end; receipts record each arm's decay in a field of its own.
7. Cost: CPU-seconds a visit for each arm on an idle host, and a forecast for release 1.

## What is recorded and computed

The definitions are D1 to D14 of the audit's section 5, entered here as the audit words them. The scorer's additions for these are made with the trial's report and compared on the trial's two lives with the reviewer's part J (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e44_l2_checks.py` on the reviewer's branch). All from the event file unless the list is named. V is the number of visits read.

- **D1. Reduction item:** an asked item of add or sub whose pair needs the reduction (`wrap` = 1; task number 4g or 4g + 1).
- **D2. Score by age on reduction items**, visit v: the share right over the reduction items of visit v, add and sub pooled. The blind predictor's: the same share of the `blind` column.
- **D3. Kinds of visit:** first visit; continuation (absence 1); return (absence 2 or more).
- **D4. Late level:** the mean of D2 over the last quarter of the visits (the last max(1, V // 4)). Reported also for each kind of visit apart.
- **D5. Score trend:** the least-squares slope of D2 on ln(v + 1) over the returns among the visits V // 2 to V − 1. The slope over all visits of that half (the memo's definition) is reported beside it and not used.
- **D6. First-visit level of a family:** the mean of add's and sub's share right on reduction items over the last 64 updates of its first visit (the E44 scorer's `lw_first_visits`).
- **D7. Trend of learning:** the least-squares slope of D6 on ln n over all families met in the V visits, n = 1 for the first met. Beside it, the coefficient of ln n in the joint fit with the logarithm of the taught pairs, as in the calibration's tables (audit, recommendation 5).
- **D8. Kept at return, from the stream:** for a continuation or a return, the reduction items of the visit's first update, which are answered before any update of the visit; right and count pooled over the visits of an absence class. Classes: 1 visit; 2 to 4; 5 to 32; 33 or more.
- **D9. Kept on the list:** for a family f answered by the network at the end of visit v, with ℓ ≤ v its most recent visit: its share right on its 200 items at v divided by its share at the end of ℓ. Tabled by v − ℓ in the classes of D8, with mean and median; a family whose share at ℓ is 0 is left out and counted.
- **D10. Relearning at return,** for the visits of D8 by absence class: the family's figure (as D6) over the first 16 updates and over the last 64 updates of the visit, beside the last 64 updates of the family's previous visit.
- **D11. Between arms:** per world the difference of a quantity; the mean and its 90% interval (t over worlds); `higher` or `lower` where the interval excludes zero, otherwise `not distinguished`.
- **D12. Closed:** the comparison of `naive` and `replay` is closed at 1,024 visits if the late-level difference is `higher` at both 256 and 1,024 visits, or `lower` at both, and at 1,024 visits neither the difference in D5 nor the difference in D7 is distinguished on the opposite side. Otherwise it is open at 1,024.
- **D13. Stuck, still learning:** as the section "How the result is stated" has them, with D5 and D7, and with the two intervals printed.
- **D14. What the wrong answers are** (working rule 3; audit, change 7): for D8's items and for the list's rows by D9's classes, the share of answers in each class of the audit's section 2.5, tried in that order, with the chance line; for `naive` and `replay`, at 256 and 1,024 visits. The classes: right; the sum or difference without the reduction; the right answer under the modulus of the family taught last; the right answer under the modulus of an earlier family of the life; the right answer under 89, the birth family's modulus; one of the operands; another two-digit number. The chance line: a modulus that the life never meets, tried in the place of the family taught last.

Also recorded per life: the absence classes restated in updates and in decay constants (4,000 updates, 15.6 visits), and the ledger (stored numbers by kind, replay share), which every receipt carries.

**The learning probe (audit, change 9; taken).** From the states after 64, 256 and 1,024 visits of the eight long lives (`--save-states`), a branch of four visits over the probe families 52 to 55 (`sw_run.py branch <state> --groups 52,53,54,55`). Reported by age and arm: each probe family's first-visit level (D6). About 22 CPU-minutes.

Lives of 64 visits are not used for any quantity over families. Committed: receipts, score tables, and event files of every life (compressed), states at visits 256 and 1,024 of world 1; hashes of the rest. While lives run, the latest state of each running life is also committed, so that a container restart loses at most what was done since; states beyond those named are removed from the branch's tree at the end (audit, recommendation 8), and the report says so.

## How the result is stated (fixed beforehand)

- **Stuck** (memo 17.4): over the second half of the longest life, neither the score trend nor the trend of learning has a 90% interval (t over worlds) on the improving side of zero. The score trend is the slope over the returns of the second half (D5). **Still learning**: at least one has. For `naive` and `replay` at 1,024 visits (four worlds), and for every arm at 256 visits (eight worlds). The two intervals are printed, and a trend is called **declining** where its interval lies wholly below zero, so that a trend not found is told apart from a trend downward (audit, recommendation 6). With four worlds the 90% interval is the mean ± 1.18 standard deviations; "stuck" is then the statement that no trend was found, at that precision.
- **Between arms** (D11): per world, the difference in late level and in kept at return (medium and long absences), with the mean and its 90% interval across worlds; *higher* or *lower* where the interval excludes zero, otherwise *not distinguished*. Comparisons: `replay` against `naive`; `naive-wd0` against `naive`; `replay-wd0` against `replay`.
- **Horizon**: every statement carries "up to V visits". **Closed (D12):** the comparison of `naive` and `replay` is closed at 1,024 visits if the late-level difference is `higher` at both 256 and 1,024 visits, or `lower` at both, and at 1,024 visits neither the difference in D5 nor the difference in D7 is distinguished on the opposite side. Otherwise it is open at 1,024, and a longer life needs a larger task space, which is a generator change and an audit.

## Release 2 (named now; released by me after release 1 is read)

The sweep and `sweep+replay` (and their decay-0 arms, if decay is read as mattering) at 256 visits on worlds 1 to 8; whatever arm the horizon rule leaves open; width, as the present width under `naive` and `replay` at 256 visits on four worlds (about 3 CPU-hours a life). A gated arm from E48 waits for a learner that produces its own gate. If `replay`'s median first-visit level in the trial is below 0.20, release 2 adds `replay` at ρ = 0.25.

## What it cannot show

- Anything about the sweep, until release 2.
- A tuned replay share or buffer, or the best decay; one value each.
- Lives of more than 52 families; anything at full scale.
- Composition, which this world does not teach.
- Decay alone beside what is kept: release 1 does not compute it (audit, 1.3 c).

## Predictions

PL2.1 to PL2.8 are the coordinator's, unchanged from revision 1; they are read with the definitions D1 to D14. PL2.3 and PL2.5 are read on absences of 33 visits or more, as written, and the same statements on absences of five visits or more are given beside them with the count of returns (audit, recommendation 4). PL2.9 is the reviewer's (audit, recommendation 7).

| | Prediction | Chance |
|---|---|---|
| PL2.1 | `naive` at 1,024 visits is *stuck* | 45% |
| PL2.2 | `replay` is *higher* than `naive` in late level at 1,024 visits | 75% |
| PL2.3 | `replay` is *higher* than `naive` in kept at return after long absences, at 256 visits | 85% |
| PL2.4 | `naive-wd0` against `naive` at 256 visits: *not distinguished* in late level | 60% |
| PL2.5 | `naive-wd0` is *higher* than `naive` in kept at return after long absences, at 256 visits | 30% |
| PL2.6 | `replay` at 1,024 visits is *still learning* | 55% |
| PL2.7 | The trend of learning under `naive` at 1,024 visits has an interval above zero | 25% |
| PL2.8 | The comparison of `naive` and `replay` closes at 1,024 visits | 70% |
| PL2.9 (the reviewer's) | `replay-wd0` against `replay` at 256 visits is *not distinguished* in late level | 60% |

**The reviewer's expectations, on record before any life** (audit, section 9). (i) `naive`'s late level on reduction items at 1,024 visits lies between 0.40 and 0.55 (55%). (ii) `replay`'s median first-visit level is below 0.20 in at least three of the eight worlds at 256 visits (50%). (iii) Under `naive`, at returns after two or more visits, more than a tenth of the answers are the right answer under the modulus of the family taught last (75%).

## Done

The two runner changes with their four checks, checks 1 to 7, the two 8-visit lives of the other two births, the trial's report, the 32 lives, the learning probe, and `E44_L2_REPORT.md` with the curves, the quantities per life, the statements above with their horizons, and the outcome of each prediction. The report lists the committed files from `git ls-tree` (working rule 4). No readings beyond these.

## What changed from revision 1

The audit's nine changes (its section 8), entered as it words them, and the decisions that the reviewer's instructions to the worker take in the coordinator's absence.

| | Change | Where |
|---|---|---|
| 1 | The second runner change is named: `load_learner` accepts the replay arms. "One small runner change" became "two small runner changes". | first paragraph; "The runner changes" |
| 2 | The birth family's 2,000 items follow both arguments. The count of answered items under the present rule is corrected (revision 1 said "about 10 million answered items per family for decay alone"). | "The runner changes" |
| 3 | `--decay-distances` takes `none`; L2 uses `--list-every 8 --decay-distances none`. The reviewer's decision: release 1 uses this. | "The runner changes" |
| 4 | The three checks of the change restated, and a fourth, (d). Check 5 compares the list file's arrays hash as well. | "Checks of the change"; check 5 |
| 5 | The recorded quantities are D1 to D14 of the audit's section 5; the sentence on closing is D12. | "What is recorded and computed"; "How the result is stated" |
| 6 | The score trend is read on the returns of the second half; the late level is reported by kind of visit. | D4, D5; "Stuck" |
| 7 | What the wrong answers are: D14, for `naive` and `replay`, at 256 and 1,024 visits. | D14 |
| 8 | What `replay` costs in new items; its median first-visit level in the trial's report; what follows if it is below 0.20. | "What the record says"; "The trial"; "Release 2" |
| 9 | The learning probe is taken. "Done" says that the report lists the committed files from `git ls-tree`. | "The learning probe"; "Done" |
| — | "check 6 measures it" corrected to "check 7". | "Runs" |
| — | Recommendation 1 (order), 2 (states and resume), 3 (the two 8-visit lives of the other births, with a stop below 0.20), 6 (the intervals printed; *declining*) and 8 (states beyond the probe's stay out) adopted as written. Recommendations 4 and 5: the stated quantities are given beside the note's own. Recommendation 7: prediction PL2.9, the reviewer's. | "Runs"; "Restarts"; D7; "Stuck"; "Predictions" |
| — | The lives without decay record the norm of each weight tensor at every eighth visit. | "Restarts" (last paragraph) |
