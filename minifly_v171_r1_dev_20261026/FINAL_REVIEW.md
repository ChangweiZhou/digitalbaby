# V171-R1 independent final review

**Scope:** Read-only source and saved-artifact review; no model queries or training by this reviewer. The parent V171 result and its source are preserved.

## Method checks

- EXACT1 truncates the saved augmented-state history to its last transition while retaining the same current parameters, latched read, target and warmup. `credit(..., mode="EXACT3")` then executes exactly one iteration because only one frame remains. Earlier `(h,r)` states are treated as constants. The independent one-step forward derivative and central finite difference qualify this conditional gradient; this is not an EXACT1-trained history.
- EXACT1/EXACT3/FULL target comparisons use cloned controllers at identical current forward state and parameters within each ON or R_ZERO condition. Gradients are accumulated at fixed weights over each answer/EOS record, then the real per-synapse clipping and parameter-box projection are applied to discarded clones. Actual steps are therefore checked rather than inferred solely from raw gradients. These one-step diagnostics do not measure the outcome of sustained training.
- R_ZERO is a separate forward lesion: setting R to zero also changes subsequent h and gate activity. It is not an algebraic identity on the ON trajectory. The `off` flag remains false, so R has a nonzero potential gradient and may regrow in the discarded applied-step diagnostic. Do not describe this as a permanently removed feedback learner.
- All 960 regenerated native/EXACT3/FULL own-output choice lists authenticate against existing saved panels; FULL teacher-prefix target log probabilities also authenticate in the runner. Original checkpoint hash receipts were absent; the new hash plus output authentication is retrospective, not an original immutable receipt.
- Independently recomputed all 115 original calibration source hashes: zero mismatches. This review does not claim to repair the nine historically lost technical arrays.

## Parameter-block credit: avoid denominator dilution

The whole-vector median ||g3−g1||/||gfull|| is 0.02743. That number is dominated by the direct gate parameters and must not be used to say all temporal parameter credit is negligible.

| Parameter block | Median ||g3−g1|| / ||gfull|| | Median ||gfull−g3|| / ||gfull|| |
|---|---:|---:|
| U | 0.595323 | 0.359247 |
| W | 0.485869 | 0.185442 |
| R | 0.466319 | 0.214937 |
| b | 0.399280 | 0.295942 |
| G | 0.000064 | 0.000033 |
| c | 0.000075 | 0.000049 |
| U/W/R/b combined | 0.446732 | 0.277342 |

For scale, the median full-gradient norm is 0.03383 in U/W/R/b versus 0.59021 in G/c; the median absolute delayed U/W/R/b difference is 0.01509. Thus ordinary recurrent temporal gradients are substantial relative to their own smaller parameter block, while contributing little to the dominant overall gate update.

The ON and R_ZERO whole-vector ratios remain almost identical (means 0.027540 and 0.027549). Together with the tiny delayed G/c component, this supports a weak *additional memory-feedback* influence in these frozen checkpoints. It does not prove absence of ordinary recurrent credit, equivalence of long-run learners, or architectural impossibility.

## Ceiling and exposure boundaries

- The joint LP audits must distinguish online pre-update tape states from frozen-checkpoint correct-prefix states. Only the latter supports the independent per-prefix oracle path envelope. An oracle with a separate target-informed gate at every prefix is not a shared controller, a learner, or evidence that the values store the answer.
- The signed-output oracle is an enlarged output intervention. Its success cannot show that V171’s positive bounded gates are feasible or that native storage is informative.
- Source and endpoint worlds remain two. All taught-item diagnostics remain E1. No held relation was opened, no new native teaching occurred and no fresh efficacy replication was performed.
- New runner work includes 960 own-query authentications, 320 teacher-prefix records/920 native target sites, 5,520 gradient target reuses and 1,920 discarded-clone parameter steps. These costs must remain separate from inherited training exposure.
- Charge both failed import/runtime attempts and the successful run. The first failed numerical attempt consumed 12 synthetic loss evaluations before assertion; the final attempt consumed another 12. Do not report only the final 12.

## Disposition

The corrective analysis addresses missing EXACT1, joint actuation feasibility, all-80 resolution and process accounting at diagnostic scope. It does not silently convert the old gate to a pass, promise local-rule success, or launch the disputed upstream/native-memory route. The next design must first settle its operational no-attention boundary and task-independent every-byte objective, and must qualify a forward graph before another temporal-rule efficacy claim.

Independent reviewer reduction/source verification process CPU: 0.031526 seconds (excluding interpreter/import startup and earlier interactive reads).
