# E44, job L2: the first long lives — roster note (17 November 2026, revision 1)

Evidence level E2. Under the pace rules of the reviewer's memo (section 17.8) a roster inside the audited protocol needs a one-page note with predictions on file and no audit; **any change to the runner is audited.** This roster needs one small runner change (below). **So the note is released for everything except that change, which waits for the reviewer's check; the lives start when it passes.** Written at the user's word of 17 November.

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

| | Worlds | Visits | Arms | Lives | CPU-hours (forecast) |
|---|---|---|---|---|---|
| 1. Trial | 1 | 1,024 | `naive`, `replay` | 2 | 8.4 |
| 2. Decay | 1 to 8 | 256 | `naive-wd0`, `replay-wd0` | 16 | 16.8 |
| 3. Long | 2, 3, 4 | 1,024 | `naive`, `replay` | 6 | 25.2 |
| 4. Short | 5 to 8 | 256 | `naive`, `replay` | 8 | 8.4 |

About 59 CPU-hours: above the memo's cap of 48 by the two decay arms, which the audit of the E44 brief recommended for this roster and E46 makes cheap to read. A life of V visits is the first V visits of a longer life, so worlds 1 to 4's lives at 1,024 give their 256-visit figures, and every arm has eight worlds at 256 visits. Forecasts are from E44's measured 0.053 CPU-seconds an update under `naive`, with 10% added for the buffer, which is unmeasured on this world; check 6 measures it.

**The trial (working rule 2).** World 1's two lives run first. The worker reports their curves and diagnostics (below) and **continues without waiting** unless a stop condition holds: a loss that is not finite; the buffer below capacity after visit 64; the score by age on reduction pairs below 0.05 over visits 512 to 1,023 in both arms; or a forecast for release 1 above 75 CPU-hours. I read the trial while the rest runs.

## The runner change (waits for the reviewer's check)

E44's runner answers the fixed list after **every** visit for every family taken so far, and on each family's first visit computes decay alone for **every** later visit of the life. In an 8-visit calibration life that is cheap; in a life of 1,024 visits with 52 families it is about 10 million answered items per family for decay alone. Two arguments, with defaults that keep E44's behaviour:

- `--list-every K`: every family taken so far is answered at the end of every K-th visit and of the last visit; the family just visited is answered at the end of every visit. Default 1. **L2 uses 8.**
- `--decay-distances d1,d2,…`: decay alone is computed at these distances in visits from a family's first visit only. Default: every later visit. **L2 uses 1, 2, 4, 8, 16, 32, 64, 128, 256, 512.**

**Checks of the change.** (a) With the defaults, world 1's 8-visit calibration life at 256 updates reproduces E44's committed list file and receipt hashes exactly. (b) A 16-visit life of world 1 with `--list-every 4 --decay-distances 1,2,4,8` gives list rows that are each equal to the row of the same visit, family and distance in the same life run with the defaults. (c) The event file and weight hashes are the same with and without the change (the list does not learn).

## Checks before the lives

1. The E44 runner at the commit used still gives check 1's digests (worlds 1 to 8, schedules and tasks).
2. The births of seeds 93683 and 116089: file hashes equal their E41 receipts, and A and R through the runner equal the receipts' final rows (as E44's check 3).
3. `replay` with ρ = 0 equals `naive` on world 1 for one visit: same weights hash and event file.
4. The buffer's capacity printed at the start is 55,209 items, and it is never exceeded.
5. Resume: a `replay-wd0` life of world 1, stopped at visit 4 of 8 and resumed, gives the uninterrupted hashes and event file.
6. The decay arms read back `weight_decay` 0.0 in the optimizer's first group at the start and the end; receipts record each arm's decay in a field of its own.
7. Cost: CPU-seconds a visit for each arm on an idle host, and a forecast for release 1.

## What is recorded and computed (the E44 scorer; read on reduction pairs)

Per life, from the event file: **score by age** on asked reduction items per visit, with the blind predictor's beside it; **late level**, the mean of score by age over the last quarter of the visits; **score trend**, the least-squares slope of score by age against ln(visit + 1) over the second half; **trend of learning**, the slope of first-visit level against ln n over every family met, n from 1; **kept at return**, the share right on the return's first update by absence class, with absences of one visit tabled apart from 2 to 4, and the absence classes restated in updates and in decay constants (4,000 updates, 15.6 visits: short 0.06 to 0.26, medium 0.32 to 2.0, long 2.1 or more); **relearning at return**; the list at the stated visits with decay alone beside it; and the ledger (stored numbers by kind, replay share).

Lives of 64 visits are not used for any quantity over families. Committed: receipts, score tables, and event files of every life (compressed), states at visits 256 and 1,024 of world 1; hashes of the rest.

## How the result is stated (fixed beforehand)

- **Stuck** (memo 17.4): over the second half of the longest life, neither the score trend nor the trend of learning has a 90% interval (t over worlds) on the improving side of zero. **Still learning**: at least one has. For `naive` and `replay` at 1,024 visits (four worlds), and for every arm at 256 visits (eight worlds).
- **Between arms**: per world, the difference in late level and in kept at return (medium and long absences), with the mean and its 90% interval across worlds; *higher* or *lower* where the interval excludes zero, otherwise *not distinguished*. Comparisons: `replay` against `naive`; `naive-wd0` against `naive`; `replay-wd0` against `replay`.
- **Horizon**: every statement carries "up to V visits". A comparison of `naive` and `replay` is closed when the sign of the late-level difference is the same at 256 and 1,024 visits and the trend difference does not point the other way; otherwise it is open at 1,024, and a longer life needs a larger task space, which is a generator change and an audit.

## Release 2 (named now; released by me after release 1 is read)

The sweep and `sweep+replay` (and their decay-0 arms, if decay is read as mattering) at 256 visits on worlds 1 to 8; whatever arm the horizon rule leaves open; width, as the present width under `naive` and `replay` at 256 visits on four worlds (about 3 CPU-hours a life). A gated arm from E48 waits for a learner that produces its own gate.

## What it cannot show

- Anything about the sweep, until release 2.
- A tuned replay share or buffer, or the best decay; one value each.
- Lives of more than 52 families; anything at full scale.
- Composition, which this world does not teach.

## Predictions (mine)

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

## Done

The runner change with its three checks (after the reviewer's check), checks 1 to 7, the trial's report, the 32 lives, and `E44_L2_REPORT.md` with the curves, the quantities per life, the statements above with their horizons, and the outcome of each prediction. No readings beyond these.
