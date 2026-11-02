# Dogfood T0: the kit applied to the completed LEARNED-BANK V1 delivery

24 October 2026. This is the first use of the research-loop kit on a real, completed delivery: LEARNED-BANK V1,
`origin/claude/charming-clarke-u9g850` at `656228be`, reported as "partial support, 3 of 4 worlds".

It ran at tier T0: no learner, store, surrogate or delivery code was executed. Two agents worked from saved per-prompt
outputs with the stable checker `0471fda3`, extracted with `git archive`. Machine record:
[DOGFOOD_LEARNED_BANK_20261024.json](DOGFOOD_LEARNED_BANK_20261024.json). Scripts and outputs stayed in the container's
`/tmp`.

## 1. Independent review, following `minifly-independent-review`

**Status: REVIEW_DEBT, not a Tier-2 review.** The reviewer is a Claude subagent and the delivery was written by Claude,
so the skill classes this review as review debt. The cross-family (Codex) review is still owed.

Executed checks:
- **Numbers are honest.** Every headline number reproduces from the raw per-prompt outputs with independent code: the
  per-world, per-arm balanced held accuracy, the margins, the alignment rises, the gates and the 3/4 label (within 5e-5).
- **The decision code fails open, and in a direction that can upgrade the verdict.** On mutated inputs, the locked
  `summarize.py` would turn the verdict from "partial" into "supported" (3/3) if the failing world's SHUF history were
  missing. It also tolerates a missing ORACLE, a missing T-COPY run, one world, dev-world names and an extra world. The
  delivery's posthoc check rejects all of these. This run's outcome is unaffected: all 32 locked histories are present.
- **The fly's own answer does not beat the shuffled-label control.** On the system answer, the score that actually
  leaves the fly, LEARN minus SHUF is -0.108, +0.244, +0.125 and +0.143. The 0.20 margin holds in 1 of 4 worlds. The
  pre-registered metric is the learned bank's component readout, so the locked verdict stands, but it is a component
  result. The REPORT's system-answer table omits SHUF.
- **Cheapest shortcut.** No non-learning policy reaches the locked rule: the best ladder or lookup policy scores at most
  0.63 balanced. On a binary one-byte answer, though, the addition-shaped ladder is nearly degenerate.
- **Level actually earned: none for the pre-registered primary.** "Partial" is not "supported", and even a pass would
  be a component readout capped at E1, with no own EOS and no no_prior_teaching arm. Under the kit's own aggregators,
  `all_worlds` FAILs, `k_of_n 3/4` PASSes, and the paired LCB is 0.140 against a 0.20 margin.
- **Retro and lint** flag lock timing, the closure gap, fail-open code, claim wording and the missing AUTHORIZATION. The
  stable CLM-06 misses "partial support ... by the locked rule". It only flags "partial" in a sentence that also
  contains "gate".

## 2. Schema-fit dry run: the delivery re-rendered as a schema-v1 package in `/tmp`

**Under the stable kit, LEARNED-BANK V1 would not have been allowed to run.** The lock gate FAILs: 4 worlds against the
8-world floor, no predictions, no recorded options, and world ids that collide with other files. In the true
chronology, AMD-02 also FAILs: the task changed from T-MOD to T-PAR after learner outcomes, without an owner decision.

The 33 friction items fall into six kinds:

| Kind | Count | Examples |
|---|---|---|
| true finding: the kit is right | 7 | too few worlds for MAIN; caps not in the owner's words; no cross-family review before spend (the run started 1.7 min after the lock); post-outcome task switch; a lock closure through a gitignored zip; output collapse flagged by LAD-04 |
| false positive: legitimate design wrongly blocked | 7 | INERT alarm when no arm can have own EOS; ADM-EXPO counts prompts held in one world and taught in another; an OLD pool is mandatory; world ids M1-M4 token-match unrelated files; CLM-07 objects to the program's own word "native"; AUT-02 looks for the substring "official"; arithmetic heuristics fire on config constants |
| schema gap: the kit cannot express the design | 11 | no evaluator-length scoring for one-byte answers; FIXED_1 equals the candidate on fixed-length answers; one held manifest for all worlds; outcome gates (ORACLE >= 0.9) not expressible; hyperparameters passed as CLI flags escape designs_tried_k; worlds passed by environment variable escape the rehearsal check; no readout field (sub-bank vs system answer); the generalization type is conflated with the task family |
| missing runner data | 4 | no admit.py, receipts or ADMISSION; no launch records; testbed logs kept only as aggregates |
| ceremony cost | 3 | 54 `__FILL__` lock fields; prediction format; headline and finding edits that need several claim reruns |
| skill unclear | 1 | how cycle 0 runs when the input code lives inside the learner module; non-addition component endpoints |

**Estimated adoption cost** for a delivery like this one: about 2-3 agent-days plus half a reviewer-day, against about
1 day as actually run. The checker itself is cheap: gates take 1-3 s and selftest about 32 s. Most of the cost is the
runner contract and admit.py, which are written once per learner family and then reused.

## 3. What this means

- **The kit catches real problems.** The decision code that could upgrade a verdict, the component-versus-system gap,
  the missing review, the task switch and the small world count are each caught or surfaced in seconds.
- **The kit is addition-shaped.** It cannot yet accept legitimate component or non-addition experiments without false
  positives or fake fields. An agent facing that is pushed to game the kit, which is the opposite of the goal. Making
  it accept these experiments is the main work of checker 1.0.2, together with the LEARNED-BANK session's review
  (`reviews_20261024/MRK1_KIT_0471fda3.md`, 16 items, one BLOCKER).
- **Proposed 1.0.2 scope, in priority order:**
  1. the testbed ceiling: exclude ORACLE_FEATURES, use a ceiling grid, require at least 2 worlds;
  2. a component metric for evaluator-length answers, and FIXED_k excluded on constant-length pools;
  3. per-world split manifests and per-world exposure checks;
  4. an optional OLD pool;
  5. a `diagnostic` launch stage, so surrogate and screen runs count toward designs_tried_k;
  6. a design identity that includes CLI hyperparameters;
  7. caps requested at S0 together with a measured cost;
  8. fix the false-positive scanners (world ids, "native", "official", config constants);
  9. comparator noise: a per-arm paired LCB option, and a SHUF band computed from the label marginals;
  10. an `import-legacy-run` command for labelled legacy renders.
