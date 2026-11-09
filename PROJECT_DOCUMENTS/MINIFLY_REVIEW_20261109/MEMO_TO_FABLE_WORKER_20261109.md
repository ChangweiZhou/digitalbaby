# Memo to the main Fable worker: three answers from the project side, and what the 9 November review found

- **From:** the cloud review session of 9 November 2026 (the session that delivered the research-loop kit on 24 October).
- **Reviewed:** `codex/minifly-consolidated-20261023` at `c8cfbbbc` and `claude/vibrant-gauss-tops5f` at `ac98a824`.
- **Method:** read-only. I ran no experiment and imported no harness code. Scores are recomputed from the committed receipts, and weights were read with a torch-free reader. Section 8 lists three scripts that reproduce the main claims. A figure marked *(relayed)* comes from a sub-reader and I did not re-derive it.

## 1. Summary

- **Three questions my review ended on have been answered.** The answers are quoted in section 2. The second one probably puts the life rule used since Experiment 1 outside what is allowed, so it needs a ruling before more lives are launched.
- **E17's local arms have a defect.** They erase the input embeddings of every operator, distractor letter and task code. The Phase 1 and 1b readings are confounded by it, and Phase 1c inherits it (section 3).
- **The Phase A gate counts are right as pre-registered, and weaker when the items are separated.** 18 of 21 lives pass on family means; 4 of 21 pass on the add/subtract compositions alone (section 4).
- **What I think stands regardless:** the diagnosis in your review of 6 November, the fact that a shared-computation network generalizes on mod-97 where the hash memory never did, replay coverage as the lever for retention, the runaway of self-generated replay targets, and the practice of writing predictions first and reporting misses.

## 2. Three answers from the project side

They were given on 9 November by the person directing the review session and are quoted whole. The readings under each one are mine, not the project's. I cannot tell from here whether the words are Mr. Zhou's own, so please confirm with him before filing them as his decisions.

### 2.1 What "mini" means

> **Q.** Is a small ordinary network still the project, and does "mini" still apply (37k parameters on 2 Nov, 800k now)?
>
> **A.** "Mini is relative, but let us say from first principle perspective there is nothing inhibiting us building a model with only Macbook Pro level's compute to reach baby level intelligence. This was the motivation of the original digital baby project. So we essentially step back from AGI and ask what we can achieve with limited but still significant compute power. After all human brain only use very little joules per hour."

How I read it:

- "Mini" is a compute budget, one MacBook Pro, and not a parameter cap. The 800,081-parameter E16 learner is inside it. So is a much larger one.
- The goal is baby-level intelligence at that budget. The program has no definition of that yet. The Phase B battery is the nearest thing, which makes it the definition of the goal and not a later stage.
- Results should be stated as capability at a cost: CPU-seconds and memory on the reference machine beside every score.
- The answer does not name fly structure as a requirement. Answer 2.3 still uses the fly as the standard for what learning should cost.

### 2.2 What may propagate

> **Q.** Does a backward pass using weight signs count as local?
>
> **A.** "Depends on what you call as backward pass. In general, you can propagate info in whatever way you like, but full circuit propagation is not allowed."

How I read it: any route is allowed except one that runs through the whole circuit.

What the life rule does in code: every linear map is `SignLinear` (`exp3_state.py:91-104`), `exp11_cell.update` calls `loss.backward()`, and the output error is relayed back through `l3`, `l2` and `lm`, the state, `lp` and `l1` to `book`. It is cut in time upstream of the cell (`h.detach()`), not in depth. On the plain reading of the answer, that is propagation through the full circuit, with signs in place of weights.

I did not get a ruling on the boundary, and it should not rest on my reading. Six concrete cases to put to Mr. Zhou:

| Case | What travels where | Where the program uses it | Plain reading |
|---|---|---|---|
| A | The output error relayed layer by layer through every layer to the input, using weight signs | every "local rule" life from Experiment 1 to E17 | not allowed |
| B | The output error sent to each layer directly, in one hop, through fixed or learned feedback | the V181 rule; in this harness, run on mod-97 addition only | allowed |
| C | A signal computed inside a layer, or passed between immediate neighbours only | not run | allowed |
| D | Full backprop during development (the birth), through time where there is a state | every birth since Experiment 1 | unclear |
| E | The trace computed by a backward scan in time (`trace_grads_scan`), equal to a forward per-synapse trace | every E16 and E17 life | unclear |
| F | Neighbour-to-neighbour messages iterated until they have crossed the network | not run | unclear |

If case A is ruled out:

- Every "local rule" result since Experiment 1 was obtained with a rule outside the line. The record should say "sign-symmetric feedback through every layer" where it says "local".
- The evidence for a rule inside the line is thin. On mod-97 addition at half the pairs, direct feedback peaks at 75–83% and ends at 57–74%, against 95–98% for the sign rule. At 15% of pairs it reaches 36–39% against 61–81% (your sections 9.1 and 9.2). A scalar-reward rule learned nothing (2–8%, Experiment 1). Holding, composition and retention have never been run under either.
- The program's question becomes how much of P1 to P4 survives when the error may not be relayed through the circuit.

If case D is out as well, every birth has to be redone under an allowed rule or from a random start. The mod-97 direct-feedback result above was from a random start.

### 2.3 What replay may cost

> **Q.** Is replay of stored labeled examples acceptable as the memory?
>
> **A.** "Replay of stored labeled examples is fine, but seems quite expensive. The fly does not seem to rely on replay of stored predator examples to avoid the fly swatter, our hand or other devices. So computation complexity has to be kept in mind."

How I read it: replay is allowed and its cost counts. The example sets a harder standard than anything measured so far, which is one exposure and no rehearsal.

What replay costs in E16 today:

- Half of every batch is replay, so each new item costs twice the compute.
- The reservoir has 65,536 slots against 38,736 distinct taught items (9,408 + 24,000 + 5,328). At the end of a life it holds about 95% of the items of the two operation families and about 55% of the compositions.
- Each distinct item is presented as new about 63, 16 and 74 times in the three families, before any replay.
- A learner life takes about 2,150 CPU-seconds on the owner's machine and 5,000 on the cloud container.

So Phase A shows retention with a memory larger than the world and dozens of presentations per item. It says nothing yet about the case the answer describes.

## 3. E17: the local arms erase their own input embeddings

**What the weights show.** In `exp45/e17_life_T_bank2_raw65536_<seed>.pt`, every element of the `book.weight` rows for the operators, the letters A–Z, the birth codes `p q v w` and the life codes `r t x n s u y z c` equals the birth value times 9.9387e-5. That factor is weight decay alone over 36,864 updates. Those rows received zero gradient for the whole life.

| Row group, mean norm, seed 73129 | Birth | Local learner (Phase 1) | Frozen first bank (Phase 1b) | Backprop twin |
|---|---:|---:|---:|---:|
| Operators `+ - > <` | 0.362 | 0.00004 | 0.00004 | 0.449 |
| Letters A–Z | 0.246 | 0.00002 | 0.00002 | 0.357 |
| Life task codes | 0.0016 | 0.0000002 | 0.0000002 | 0.304 |
| Digits | 0.459 | 0.077 | 0.480 | 0.373 |

Digits are shown for contrast: they are inputs at answer positions and do receive gradient. Seed 93683 shows the same factor in Phase 1. The windowed E16 learner does not have the problem (seed 900007: operators 0.38 to 0.50, life codes 0.002 to 0.62).

**Why.** In `arch17.forward` the input is the current byte only. With `through_time=False` the previous state is detached, so the loss at an answer position reaches `book`, `l1` and `lp` only through the byte that is the input at that position: `=`, a digit or `;`. The trace gives credit through time to `lc`, `lw`, `lc2` and `lw2` and to nothing upstream of them. AdamW decays every parameter in its group at every step, gradient or not.

**What follows.**

- The learner cannot tell which operation or which task is asked. That is enough to explain the collapse, so Phase 1 is not evidence about sign feedback or about stacked banks.
- `E16_FREEZE=bank1` freezes `l1.`, `lp.`, `lc.` and `lw.` and leaves `book` in the decayed group. The operator rows fall to 21.5% of birth after one block and 4.6% after two. That schedule fits "acquired in block 1, lost in block 2" without the onset of replay. This step is my inference.
- Phase 1c inherits the defect. `E16_FEEDBACK=exact` changes the feedback path and keeps the detach, while arm M uses backprop through time and reaches every row. "Exact feedback fails, M holds" would then send P17.1c.3 to the oracle arm for the wrong reason.
- `e17_probe_state.py` reports the relative move of `book.weight` as a whole, which hides the per-row pattern.

**Controls.** Exclude `book` from weight decay, or freeze it in life, and rerun Phase 1 and 1b. For any new architecture, add one qualification step: after a dry run under arm T, list every plastic element whose accumulated gradient is exactly zero.

**The part that is not a bug.** With credit cut at the state, this rule cannot learn an input code for a symbol that matters only through memory. The life codes start near zero, because the birth's own decay erased the unused rows, and nothing can raise them. The window hid this in E16, since every prompt symbol can sit in the input at an answer position. A windowless learner needs a signal from the state back into the input pathway, and whether one is allowed is the question of section 2.2.

## 4. Phase A, with the items separated

The gate counts in the record are correct as pre-registered. The same 21 raw-reservoir lives, scored four ways:

| Scoring at the last checkpoint | Staged, 9 lives | Drift, 12 lives | All 21 |
|---|---:|---:|---:|
| As pre-registered: three family means, compositions by final answer | 8 | 10 | 18 |
| Compositions by the whole worked answer | 6 | 8 | 14 |
| Add and subtract mod 97 only, k = 12 | 7 | 8 | 15 |
| The `+-` and `-+` pairs only | 1 | 3 | 4 |

- **Two of the four held pairs are easy.** On `><` and `<>` a rule that ignores the first operator scores about 70%, and the network scores 15–60% at birth. The learner ends at 89.5% against 98.7% for the reference. On `+-` and `-+`, where the ceilings are about 1%, it ends at 64.5% against 86%.
- **The gate was not met on the hard pairs on the day it was declared.** In E16-R3 the `+-` and `-+` ratios were 0.796, 0.686 and 0.769.
- **The reference is a ceiling.** Arm D draws 64 fresh labeled items per batch from every family seen so far, in equal shares (`joint_stream`). The "drift twin" does the same and never calls the drift sampler; its block losses equal the staged twin's at checkpoints 1 and 2 in all nine seeds that have both. It also runs at decay 0.3 against 0.25. Text that says "a twin trained on the same stream" is not accurate.
- **The matched control is arm M.** With a reservoir byte-identical to the failing learner's, it reaches 0.97 and 0.95 of the ceiling (E16-M2). That is the evidence that the rule loses the points. It exists for the two failing seeds, and for the three standing seeds only under the old uniform replay (E16-D2).
- **Much of the held material is taught in another form.** Of the sealed mod-97 panel pairs, 40–52% are also the first step of a taught composition, with the intermediate supervised. Of the held compositions, 47% have their first step inside a taught composition, 44% their second, and 21% both.
- **Pass counts move with the platform.** Receipts are bit-identical between two cloud containers. Between the owner's machine and the cloud the four-pair mean moves by about 3 points, and five passing lives are within 3 points of the line *(relayed)*.

## 5. Smaller corrections to the record

1. **E16-L.** "Does not acquire addition or subtraction at all" is not what the receipts show. After block 1, add/sub stand at 47/40, 30/23 and 22/14. They fall to 3–12 after block 2 and stay under 6. In two seeds that is far below the birth's own add score of 39–40.
2. **Other prose that differs from the receipts** *(relayed)*.
   - E16-RC "flat from 16,384 up": the three-seed composition mean rises 74.2, 76.9, 78.3, 79.8.
   - "`+-` moves up to 16 points between two runs": 9.3 points on identical panels, and the rest is the panel change.
   - E17 "the frozen twin holds 93 throughout": it ranges from 89.4 to 95.0.
3. **The panel collision is fixed in one file only.** `seed + 1` still keys the held panels in `exp7_decon.py:131`, `exp8_life.py:150`, `exp8_e16w2.py:150`, `exp8_e16wr.py:159`, `exp8_e16wr2.py:181` and `exp10_ring.py:161`.
4. **E16-F6 was run twice.** `claude/vibrant-gauss-tops5f` holds the complete run and its report (5 of 6 at the gate; P16F6.5 misses at rho 0.217). The default branch holds a second, partial run by another worker. One of them should be named the record.
5. **Predictions.** Of 33 numbered predictions in the nine cloud experiments, 17 held as written. Of the 12 about a mechanism, an equivalence band or a signature, 1 did. P16F6.5 was already contradicted by the six drift seeds in the repository when it was written *(relayed)*. Checking each new prediction against data already on file would catch that.
6. **History.**
   - `codex/minifly-three-experiments-20261106` was never merged. It holds the original mod-97 run your section 9 rests on. Its audit gives full minus shared-only as +4.2, +2.5 and −4.0 points by seed, so "about one percent" is the KC-alone score *(relayed)*.
   - `codex/minifly-whole-brain-learning-scope-20261106` holds a note correcting the statement that the central complex was never in the model *(relayed)*.
   - V186 ran the 3 November sensor-decay proposal that your section 5 says was never run: +0.54 bits, still worse than a bigram *(relayed)*.
7. **The kit.** `AGENTS.md` and `SKILL_READINESS_20261102.md` say the checker's dispatcher and core modules are missing at tree `102abf4b`. They are present there. I am not asking you to adopt it: it bans `.backward(` and is built around the old addition task.

## 6. What I would do next

These are proposals. The order matters more than the details.

1. **Get the ruling on section 2.2 first.** Put the six cases to Mr. Zhou and write his answer down verbatim. Hold new lives under the sign rule until then. If case A is out, they are not worth the machine time.
2. **Repair the record without compute.** Run the embedding-row check on the existing E17 weights, mark the Phase 1, 1b and 1c local-arm readings as confounded, call arm D a ceiling, and apply the corrections of section 5.
3. **If case A is out, climb the existing ladder again with a rule inside the line.** Start with the smallest rung: mod-97 is done, then holding (E11), composition (E7), and the E16 stream last. Run each against the sign rule and arm M, and stop at the first rung that fails. That rung is the research question. Candidates plainly inside the line are direct feedback with fixed or learned weights, and objectives local to a layer.
4. **Put cost in every receipt.** Record CPU-seconds on the reference machine, presentations per distinct item, replay share, and reservoir slots and bytes. Then run the one curve the cell learner has never had, which is retention against replay share at a fixed reservoir.
5. **Design the one-exposure test the swatter example describes.** One presentation, no rehearsal, scored after intervening learning, with compute per event reported. It is the task on which a sparse fast memory read at recall has a fair test. The exact-key store was retired on a task that never needed it: it never entered training and could not match an unseen prompt.
6. **Write the Phase B battery now, with its compute budget, as the definition of "baby-level".** Keep the arithmetic stream as a regression suite.
7. **Change three reporting habits.**
   - Report the hard items separately, as gain over the best shortcut and over the birth.
   - Judge the rule against arm M, not the ceiling.
   - Use a criterion over the distribution of seeds, fixed in advance, in place of pass counts at 0.80.

## 7. What I did not check

- Codex's window-net harnesses (E16-W, WR, WR2, WL) and their reports.
- The E17 weights of seed 116089, and of 93683 in Phase 1b.
- The same-weights panel scores in `E16RP_SAME_WEIGHTS.json`, which need torch.

## 8. How to re-run the checks

All three scripts are stdlib only and read-only. They sit beside this memo in `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/` on `claude/vibrant-cerf-sje726`. From another checkout, fetch one with `git show origin/claude/vibrant-cerf-sje726:PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/<name>`.

- `python3 -I check_e17_embedding_rows.py BIRTH.pt FINAL.pt` prints the row norms and the decay factor of section 3. For the table there, the pair is `e17_dev_bank2_73129.pt` and `e17_life_T_bank2_raw65536_73129.pt` in `MOD97_LOCAL_RULE_20261106/exp45/`. Any birth and end-of-life pair works.
- `python3 -I gate_by_item_type.py [REF]` prints the per-seed table behind section 4. It reads receipts through `git show`, and `REF` defaults to `origin/claude/vibrant-gauss-tops5f`.
- `python3 -I exposure_overlap.py [SEED ...]` re-derives the splits and prints the overlap figures of section 4.
