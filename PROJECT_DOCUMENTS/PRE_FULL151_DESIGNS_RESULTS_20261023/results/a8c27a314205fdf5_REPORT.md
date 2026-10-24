# Step 10 — V9.9A value memory driving an emitted two-option choice

**Result (27 September 2026): the narrow bridge passed.** On 64 fresh confirmation worlds, the frozen single-store REF/FE0 learner and fixed sequential choice organ emitted the correct `L/R` byte for **68.75%** of never-taught pairs after old teaching, a day, disjoint new teaching, and another day. With the old native writes disabled, accuracy was **46.88%**. The world-paired W−N difference was **+21.88 percentage points**, with a two-sided 95% Student-*t* interval of **+13.59 to +30.16 points**. All prespecified trained-cue and absolute held-out guards passed; the independent reducer verified the `LIMITED_E3_EMITTED_CHOICE_SUPPORTED` verdict.

This is an **output-interface demonstration and fresh-world replication of V9.9A's limited within-family E3 signal**, not a newly discovered memory mechanism. The environment supplies both alternatives and a response marker, and an engineered fixed comparator converts native values to a byte. The fly does not generate an answer candidate, decide when to respond, or answer a single open question such as `1+2=`. This artificial choice setting is useful for localization but is not yet the intended real-life behavior.

## What was learned, presented, and emitted

The [prospective design](DESIGN.md) inherited the V9.8 E3 generator and V9.9A REF learner without modifying their code or checkpoint. Each world taught six P–N symbol edges in both orientations over 144 records, withheld three edges in both orientations, then delivered 72 records on disjoint new symbols between two 24-hour gaps. No held-out ordered pair or answer was ever reinforced. Old W and N received identical bytes, labels, teaching calls and clocks; only W allowed old native value writes. Both allowed new writes. Exact-byte and canonicalized-pair recitation tables have no held-out answer; order-balanced presentation makes a constant `L/R` response score 50%. A learned one-bit symbol-class rule can solve this family, so E3 here is narrow.

At each read-only probe, the fixed [choice organ](choice_organ.py) consumed `12 cue bytes | 12 cue bytes ?`. It stored the first cue's native value, read the second cue at its later presentation time from the same learner snapshot, and emitted the side with greater value. It received no label, hidden class, pair identifier or correct-side flag. The scorer examined **only the emitted byte after the response marker**. Each pair was presented in both orders on isolated probes. The organ's separator reset and one-scalar trace are engineered interface assumptions; they do not arise from learned Full151 circuitry. The second cue's later time also means these values are not numerically identical to V9.9A's simultaneous evaluator probes.

## Frozen gates and confirmation

Technical world 12200 passed, including exact agreement of each option value with an independent matched-time separate-cue reference and state-nonmutation checks. Pilot worlds 12201–12216 computed **trained-pair outputs only**, passed all five prespecified positive-control gates, and were excluded from confirmation. The source lock was frozen before the technical world; confirmation used precisely worlds 12301–12364 with no replacement or optional stopping.

| Confirmation output accuracy | W old writes on | N old writes off |
|---|---:|---:|
| Old trained pairs immediately after teaching | 92.58% | 46.48% |
| Old trained pairs at final probe | 89.58% | 47.40% |
| New trained pairs immediately after new teaching | 82.81% | 86.98% |
| Never-taught pairs immediately after old teaching | 71.35% | 47.66% |
| **Never-taught pairs at final probe** | **68.75%** | **46.88%** |

The final old-trained W−N difference was **+42.19 points** [36.59, 47.79]. The **sole confirmatory primary** was final held-out W−N: **+21.88 points** [13.59, 30.16], calculated over 64 independent world differences, not 384 individual option presentations. It was positive in 38 worlds, tied in 20 and negative in 6. Final held-out W accuracy passed the locked 60% floor. Accuracy was identical for correct-side `L` and `R` in each branch (W 68.75% for both; N 46.88% for both), consistent with the balanced two-order scoring. Earlier stages and side checks are descriptive, not additional confirmatory endpoints.

## Meaning for the Step 9 failure

This run shows that V9.9A's native value signal can reach a fixed byte-emitting choice pathway. It therefore rules out a **universal** claim that Full151 value memory cannot drive output. It does **not** repair Step 9: Step 9 used fading-pair input, two opponent-taught output stores and a single-cue absolute `0/1` decision, and its held-out values were near chance immediately after teaching. The two runs cannot isolate whether the lost transfer came from the conjunctive input features, two-store teaching geometry, or the absolute decision task.

The next attribution step should be small: replace Step 9's fading-pair front end with **FE0 alone while keeping its two-store teacher, single-cue output and chronology fixed**, on fresh worlds with a trained-cue gate. If transfer returns, the added pair features are implicated; if it does not, they cannot be the sole cause. Role-separated features or a broader four-arm search should wait for that result. Separately, moving toward ordinary use still requires one-cue responses, capacity under many facts, learned stopping and multi-byte output; this forced-choice bridge cannot stand in for those tests.

## Integrity and resources

The [source lock](results/SOURCE_LOCK.json) digest is `38050b3628c1fb5ade2e2753d511fbe4a74a5b97059f43ae7cadc2cc32992c8a`. The [pilot](results/PILOT.json), [confirmation summary](results/SUMMARY.json) and [independent audit](results/AUDIT.json) are accompanied by per-world receipts under `results/pilot/` and `results/worlds/`. The independent auditor regenerated the source fixture and probe roster, verified W/N teaching and clock ledgers, each option order, emitted-byte arithmetic, source hashes and all reductions, and rejected 15 deliberate receipt mutations. It did not rerun learner trajectories or independently recompute the native values of every science world. There is no `FAILURE.json`. Confirmation learner runtime summed to 1,136 seconds; receipts occupy about 10 MB. No V9.8, V9.9A or Steps 7–9 completed world was rerun.
