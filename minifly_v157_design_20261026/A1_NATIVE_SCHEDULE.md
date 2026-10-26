# V157 A1 — fixed-dose native feedback order

**Design only.** This is a separately authorizable native experiment, not an extension of V156's completed execution. All failed attempts and technical observations share its caps. No cached V156 trajectory is rerun or counted as fresh.

## Fixed learner and data

- Executable parent: V154 `native154.Native('N_2B','P_SLOW')`, inheriting V152's native signed teacher and Full151 stores. Birth, anatomy, codebook and all dependencies must match the inherited source lock.
- Sixteen independent mutable native compartments; no added answer head, q=0, homeostasis off, fresh adaptation, unchanged P_SLOW, all 257 classes, lowest-index argmax ties, at most four own decisions. A cap is truncation, not EOS.
- Update command remains `a=0.2*s(n)*P*C^T*(p-e_y)/4`, with the inherited native signed transition, not a fitted gradient through an evaluator. `P=I-11^T/16` and `s(n)=1/16+(15/16)min(n/1024,1)`.
- Reserve worlds **202610265701 and 202610265702**. Before execution, check the experiment ledger for reuse; a collision blocks the design rather than silently replacing a seed. No data or trajectories are generated for this delivery.
- Reuse V154's external generator and its `V154|world|stream` randomization convention unchanged: operands 0–19, unordered commutative orbits, one fixed orientation per orbit, 64 OLD, 64 potential NEW, 32 excluded validation and 50 assessment orbits. The evaluator renders raw prompts and expected sums; none of those semantics enter the learner.
- Rank potential NEW IDs by `SHA256('V157|'+world+'|NEW|'+record_id)` with digest-byte order and ID tie-break. Select the first 16. Use ascending order in world 5701 and its reverse in 5702. The other 48 potential NEW and 32 validation records are excluded from this run.
- Lock raw/split hashes, OLD orders from the parent generator, selected IDs and schedule lists before any model query. Hash and audit assessment membership without supplying its labels to a learner. No overlap by raw key or operand orbit is allowed.

Every scientific answer has one or two payload bytes followed by EOS. The learner has no answer-length input or digit-only class mask; all 256 bytes remain legal payload. Technical fixtures may use arbitrary bytes, with at most two payload bytes per observed record to preserve the target cap.

## Histories and treatment

Per world, prepare two fresh identical native births. W receives eight ordinary OLD passes with writes enabled; SH_OLD receives the same prompts, expected replies, EOS and timing with OLD value writes inhibited. The latter retains the real observation counter and nonplastic evolution. Both have 512 OLD record applications and n≥1024 before NEW. This does not test acquisition from scalar-only feedback or early-warmup scheduling.

Clone each OLD endpoint into two NEW branches. INTERLEAVED uses `order × 8`; CLUSTERED uses `each record × 8` before moving to the next. Enable NEW writes in all four branches. No branch is resampled because it has no initially correct OLD answers.

The two schedules are equal in final physical records, unique cues, per-cue presentations, revealed target/byte multiset and native time. At intermediate counts they are equal only in record count. In particular, at 16 records clustering has taught two cues eight times and interleaving all 16 once. Save per-cue exposure counts, expected-target counts, input/reply bytes and clocks at every checkpoint. Different intermediate coverage/recency is part of the treatment, not a matched-information comparison.

OLD preparations are shared physical prefixes. Their cost and exposure are not repeated four times merely because two schedule clones inherit each state. Report four OLD preparations and eight NEW continuations, all paired/correlated rather than twelve independent replications.

## Probes and causal controls

1. **Initial panels:** after OLD, score all 64 OLD + 16 selected NEW records once on each of the four physical OLD endpoints; reuse those frozen results for the identical schedule clones.
2. **Immediate acquisition:** after each of the 1,024 actual NEW updating records, query its complete own answer and all required teacher-prefix margins on independent clones. Expected prefixes must never enter the own rollout.
3. **Boundary panels:** after 16,32,64,128 NEW presentations, query all 80 records in each main branch. Count every complete payload/EOS match, first-byte/continuation/true-EOS error, CAP and each required margin.
4. **Sampled last-record controls:** immediately before the record at those same four positions, clone its parent. Execute that record with writes inhibited on one discarded fork and base-seconds idle of the same duration on another. Query both complete 80-record panels after the record. Query the **current record only** on the pre-record parent. The actual full panel is already the boundary panel: do not query it twice.
5. **Delay:** after final NEW, make a 128-byte-step idle clone and score all 80 taught records. Use `Native.rest(128)` for that delay; use `FastMemory.rest(actual_elapsed_seconds)` for the matched record-duration idle. Do not double-convert DELTA.

The sampled actual-minus-no-write contrast identifies that whole record's write effect on the panel. No-write-minus-idle identifies an observed-event effect under that reference. The pre-record current-cue query additionally supports its own matched time change. There is **no full pre-record panel** at these probes, so do not manufacture a full-panel I/O/W telescoping decomposition or claim the 32 probes exhaust cumulative interference.

Last-write-to-final deterioration also includes elapsed time and intervening observations. Call it a retention loss under the actual history; attribute competing-write damage only where matched controls establish it. If a further postacquisition interference shadow is needed, propose and count it separately rather than add it to this run.

## Native unseen scoring

Save and hash every final and delayed endpoint, freeze the input/scoring/source locks and complete the taught-history audits **globally before opening any held score**. Score all 50 assessment orbits once at each final/delayed endpoint of every branch, regardless of taught-acquisition outcome. Do not change the roster, endpoint or gate after seeing one world's score.

Held scoring is **own-only**: raw prompt, own generated bytes/EOS, then external comparison. No teacher-forced held continuation or feedback is delivered to the learner. The second delayed query is a repeated readonly evaluator query, not another independent experience. Commit the candidate's choice before the scorer compares it with the label; the scorer may hold ground truth, but must never deliver it to the candidate. Keep the candidate state immutable.

Report, separately in each world and at each endpoint:

- CLUSTERED−INTERLEAVED NEW exact counts at final equal dose, with OLD/all-NEW curves and per-position recency.
- W−SH_OLD exact held counts within each schedule. This is the effect of OLD value teaching conditional on common NEW training, not the effect of all learned experience.
- Initially correct OLD retained / initially correct OLD, alongside all 64 OLD. Empty acquired subsets are unmeasurable, not protected or failed by definition.
- Exact raw-key and evaluator-only canonical-orbit lookup, favorite complete reply, generic position-frequency, prompt-length and suffix-backoff lengths 0–4. Fit them only from each branch's actual observed history. Keep lookup coverage separate from its fallback reply. No countermodel supplies native answers.

One changed causal state control alone does not establish a broad learning core. E3 interpretation requires never-reinforced membership, native complete outputs, a relevant prior-teaching advantage and shortcut controls; the two prospective development worlds are not confirmation. An analytic adder validates data/scoring only.

## Exact observation and query ceilings

| Item | Count |
|---|---:|
| OLD updating | 1,024 records |
| OLD inhibited | 1,024 records |
| Main NEW updating | 1,024 records |
| Discarded sampled NEW inhibited | 32 records |
| **Science observations** | **3,104 records; at most 9,312 targets** |
| Technical observations | at most 16 records / 48 targets |
| **All observations** | **at most 3,120 records / 9,360 targets** |
| Matched-duration idle forks | 32; zero observations |
| Final delay forks | 8; zero observations |

All expected payload bytes and EOS count, including no-write observations. No candidate optimization or label grid is added. Native training predictions are counted separately from evaluator output distributions.

| Evaluator operation | Maximum requests |
|---|---:|
| Shared initial OLD+NEW panels: 4×80 | 320 own/TF pairs |
| Main boundary panels: 8×4×80 | 2,560 pairs |
| Matched no-write panels: 32×80 | 2,560 pairs |
| Matched idle panels: 32×80 | 2,560 pairs |
| Current-cue parent controls | 32 pairs |
| Immediate current-cue post-write scoring | 1,024 pairs |
| Final delayed taught panels: 8×80 | 640 pairs |
| **Scientific own/TF pairs** | **9,696** |
| Technical reserve | 64 pairs |
| Held final/delayed: 8×2×50 | 800 own-only generations |

Each own/TF pair has at most four own decisions + three required-prefix distributions. Held queries have at most four own decisions. Thus the ceiling is **9,760×7 + 800×4 = 71,520 evaluator distributions**. Exact repeated requests on the same immutable state may be reused by receipt, with savings recorded; the budget does not depend on those savings.

One fixed allowance includes qualification, all attempts, source/data admission, children, actual training, cloning, scoring and audit: 1,200 CPU seconds, 3,600 wall seconds, one worker/thread and 1 GiB RSS/output. CPU subcaps are 120 qualification / 900 execution+evaluation / 180 reduction+package. A charged preflight must establish conservative full-roster admission. If it does not fit, preserve its fixtures, stop before science and revise explicitly; do not remove controls or increase the cap silently. Resource limits during execution preserve completed receipts and the actual partial state without automatic replay or continuation.

## Interpretation and revision rule

No newly invented 90% accuracy or BPS threshold controls this diagnostic. Publish the fixed-dose exact counts, paired effects and all failures. If effects differ between worlds, retain that disagreement. A null does not establish equivalence or family-wide impossibility.

If complete acquisition improves under clustering but later drops, the result motivates a specifically controlled interference question; it does not automatically select availability gating. If both schedules acquire taught answers without a held causal benefit, the result remains E1 and motivates representation/computation work. If neither accumulates useful complete answers, leave native reachability and effective credit/dose unresolved and stop this schedule branch. No automatic rate controller, extra repeat, protection rule, capacity increase, assessment extension or follow-up experiment is permitted.
