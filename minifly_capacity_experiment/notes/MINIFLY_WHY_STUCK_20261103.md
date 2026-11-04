# Why MiniFly got stuck: an extended review with the recovered V36–V49 source

*2026-11-03.*
- *Sources:* `origin/codex/minifly-v39-v40-source@4a667352` (recovered V35–V49 runners) and `origin/codex/minifly-consolidated-20261023@dae98220` (now at V187).
- *Constraints:* no backpropagation and no attention (including content-addressed key–value reads). Everything else may change.
- *Process:* recursive research-loop skill S0 (evidence reconstruction) → C0 (cheap falsifiers). There is no MRK-1 checker, so no gate is claimed.
- *Evidence type:* the runs below are local development checks of historical code (1–2 seeds each). They are not confirmations.

---

## 0. A correction to my previous report

In the previous turn I wrote that the V39 superposed-store result "did not reproduce" and withdrew the hybrid recommendation. **That was my error.**
- My re-implementation built keys with `bytes(np.array_of_int64)`, which produced 48-byte keys padded with nulls. Their last 12 bytes did not identify the key.
- With the **recovered original code** (`elm_fly_capacity_v39.py`), the result reproduces:

| S2-lag-c12, 4,096 units, 205 active | retention A after D | held-out after D | report |
|---|---|---|---|
| load 32, seeds 39101/39102 | 100% / 100% | 100% / 100% | 98.4 / 93.8 |
| load 128, seeds 39101/39102 | 59% / 69% | 25% / 69% | 64.5 / 49.2 |

- Binary instead of graded KC activations gives the same load-32 result, so the precise tanh values are not essential.
- V40 (`elm_fly_complementary_v40.py`; its hash `10fa234b…` matches the report) at load 128, seed 40101: HYBRID wall **0.688**, SUPER 0.438, SLOTS 0.0, HYBRID-RANDOM 0.5. The report gave 0.596 / 0.387 / 0.094 / 0.420, so the directions reproduce.
- **Provenance caveat:** the recovered `elm_fly_capacity_v39.py` hashes to `a023736c…`, but the V39 report cites `0f9577b4…`. The file I ran is therefore a later edit of the V39 runner, not the reported one. The numbers agree anyway.

Two V40 details matter for interpreting it:
1. **HYBRID-FROZEN's perfect wall (1.0) comes with D_current 3%.** It stopped learning. The "wall" metric omits current acquisition, so the frozen arm is a diagnostic, not a candidate.
2. **HYBRID predicts the whole stream worse than SUPER** (2.37 vs 2.02 bits per byte). The headline "−0.47 bits vs SLOTS" hides this.

---

## 1. The new experiment: the "capacity wall" is a store-size knob, not a missing mechanism

V39–V44 spent five versions on arbitration, admission, utility eviction and capacity decomposition to rescue a store that collapsed above about 64 facts. **Every one of those versions held the KC layer at 4,096 units.** I varied only the unit count and the number of active units, using the recovered code with the stream, learning rule, lag front end and 12 contacts all unchanged (seed 39101):

| load (facts) | units | active K | A after D | held-out after D | one-shot E1 | D current | whole-stream bits/byte |
|---|---|---|---|---|---|---|---|
| 128 | 4,096 | 205 | 59–69% | 25–69% | 99% | 100% | 1.56 |
| 128 | 4,096 | 51 | 44% | 38% | 84% | 100% | 1.60 |
| 128 | 16,384 | 205 | 91% | 94% | 99% | 100% | 1.45 |
| 128 | 16,384 | 819 | **100%** | **100%** | 100% | 100% | 1.48 |
| 256 | 16,384 | 205 | 83% | 75% | 98% | 100% | 1.64 |
| 256 | 16,384 | 819 | 78% | 81% | 100% | 100% | 1.68 |
| 512 | 16,384 | 819 | 58% | 45% | 100% | 100% | 1.92 |
| 512 | **65,536** | 819 | **94.5%** | **89%** | 99.8% | 100% | 1.77 |

Scripts and outputs are in `minifly_review_experiments/v39_recovered_runs/`.

**Reading:**
- A single **fixed** sparse expansion with a local delta-rule readout does all of the following at once: learns new facts, keeps old ones through three interference waves, generalizes one-shot to an unseen rendering (E1 ≈ 1.0), and revises facts. It uses no backprop, no attention and no slots.
- Its capacity grows with the number of KC units: 16× the units gives roughly 8× the facts at about 90% retention.
- This is the textbook behaviour of sparse superposed associative memory: interference scales with the overlap of active sets, roughly K/N.
- The **V35 "four-way wall" is broken by a mechanism already in the archive, at a larger size.** The slot tier, arbitration and corroboration machinery were compensating for an under-sized store.

**Limits:**
- 1 seed per cell.
- Synthetic 14-byte records.
- Every byte is written (the store also learns syntax).
- The store is float64, 65,536 × 256 ≈ 134 MB. Int8 storage was free in V25, which would make it about 17 MB.
- It has not been tested on natural text.
- The scaling law (facts versus N and K) is not yet fitted; this is 8 data points.

---

## 2. Why the project got stuck — the causal chain

The table of findings below is assembled from the full archive read (Addendum 3), the recovered source and the experiment above.

1. **A capacity limit was misread as an architecture limit (Sept 9–10).**
   - V39 fixed `UNITS=4096`. Its own comment says loads were chosen because "at UNITS=4096 the store is already past the collapse point".
   - The program then added a slot tier (V40), arbitration (V41–V43) and capacity forensics (V44), instead of scaling N, the obvious one-line control.
   - The slot tier it added is content-addressed retrieval: nearest prototype by cosine, returning its value. That is exactly what the project's own rules later classed as attention-equivalent.
2. **The working byte line was abandoned for the fly-physiology line (Sept 11 onward).**
   - The fly line was V45–V88, the MiniFly iterations, FULL151 and CORE_RECOVERY. There, the tasks were random 8-of-88 PN cues with no shared structure: generalization was impossible by construction (FB1 design §4).
   - The output was a 1-bit valence through about 1,226 writable KCs. Capacity was again small: iteration 9 lost distinctions above about 24–48 associations. That is the same store-size phenomenon, re-diagnosed from scratch as "interference", with dozens of allocation, gating and evolution rounds.
3. **The biological-plausibility constraint vetoed the front end that worked.**
   - BYTE-CORE-1 measured the fading-mixture sensor (FE0) at 1.8% of an oracle's information efficiency ("FRONT_END_IS_BINDING").
   - FEAS-1 found the position-bound lag code selectable but "never adoptable" (C8).
   - The lag code is precisely what feeds the V39 store.
4. **Then the "mini" resource contract capped the model below the capacity that works.**
   - V181 limits the learner to about 37k trainable parameters and about 8 MiB.
   - The measured requirement for about 500 facts with ~90% retention is a 65,536 × 256 readout. Even quantized, that is ~16.8M weights.
   - The project then spent V164–V187 on local-credit recurrent learners, BPTT references and feature-deployment probes inside a budget orders of magnitude smaller than any store shown to retain facts.
   - **The contract and the evidence are in direct conflict, and nobody stated it.**
5. **Governance cost outran science.**
   - FEAS-1 went through 7 design versions in a day. A11 had 4 audit rounds and was never run. Post-V5 produced a FULL/LOW/v2/v3 design series. Hundreds of gates were written. Several decisive runs were voided by gate bugs (V39 G2, V53's hard-coded ALL floor, V36's unreachable fly gate).
   - The V50 self-critique already named this: "we repeatedly changed the computational object instead of maintaining one functioning reference learner."
6. **Lineages were not connected.**
   - The archive contains distinct learners: ELM (Sept 4–7); byte V20–V49; fly V50–V88; MiniFly iterations 3–30; FULL151; BYTE-CORE; and V164–V187.
   - Each restarted the substrate. Results such as "a shared error-corrective readout beats the native writer" (V9.10C, V74, V75) and "write every byte vs target only, ×7" (V52) were rediscovered rather than built upon.

**One sentence:** the project solved memorization and rendering-level representation on synthetic bytes with a sparse superposed store in early September, hit that store's capacity, and treated capacity as a mechanism problem. It then moved to substrates and resource caps that made the solution unreachable.

---

## 3. What remains genuinely unsolved (honest scope)

| Capability | Status with a scaled sparse store |
|---|---|
| E1 memorization of taught facts | Yes, on synthetic records |
| E2 retention through new learning, plus revision | Yes up to the size-limited load. Reproduced at 128 (16k units) and 512 (65k units); 1 seed each |
| Representation: new rendering of a known identity | Yes, one-shot (E1 ≈ 1.0). But it is *surface* similarity of a lag code, not semantic invariance |
| Natural text | **Untested.** The store writes every byte, so syntax-heavy text uses capacity; V9.2 showed prose needs ≥100 kB |
| E3: never-taught relations, roles, arithmetic | **Not addressed.** A fixed random expansion plus a linear readout cannot do cyclic relations (V9.10D) or compositional arithmetic. This needs structure beyond a lookup-like associative memory |
| Size efficiency | Poor: about 0.26 MB/fact in float64 at 512 facts. Major waste from writing every byte and from 256-wide outputs on all units |

---

## 4. Next steps (DESIGN_ONLY; each needs owner authorization)

Each step has a single falsifier. No step is a prerequisite for an unrelated one.

**Step 1 — Measure the capacity law (cheap, decisive for the resource contract).**
- Grid: N ∈ {4k, 16k, 65k} × K/N ∈ {0.3%, 1.25%, 5%} × load ∈ {128, 256, 512, 1024}, 4 seeds. Use int8 or float16 storage.
- Fit retention-at-90% capacity C(N, K).
- **Falsifier:** if C grows sublinearly in N with no K that restores linearity, a scaled store is not a viable route and the hybrid/slot question reopens.
- **Decision:** this tells you how large "mini" must be for a target number of facts. **Owner decision:** revise the 37k/8 MiB V181 contract, or accept a small fact capacity.

**Step 2 — Stop writing every byte, without labels.**
- V52 showed target-only writes were 7× better, but "target" is an evaluator label. Test label-free write allocation on the scaled store:
  - (a) write only when the store's own prediction is wrong (error-gated);
  - (b) recall-gated consolidation (Lindsey & Litwin-Kumar, eLife 2024): a small fast store plus a slow store written only when fast recall is already consistent;
  - (c) a count-matched random-write control.
- **Falsifier:** if (a) and (b) are not better than (c) in capacity per byte, the write-budget effect explains everything, and the class is closed as V27 closed competitive plasticity.

**Step 3 — Natural text with the scaled store as the memory, plus a separate predictor.**
- Combine a predictor (the V9.2 hashed short-context local-delta predictor, which beat Markov-2 on prose from 100 kB) with the sparse superposed store over a stationary lag code.
- Read rule: a fixed 50:50 mix, as in V43's FIXED-50. This is not content-addressed: both are fixed-code linear readouts.
- Probe embedded name→attribute facts, later corrections, and held-out phrasings. Score real own-byte EOS answers on deterministic probes only.
- **Falsifier:** if fact retention on text collapses at loads that the synthetic run sustains, syntax/statistics crowding (the V35 O1 diagnosis) is the binding problem. The next lever is then write allocation (Step 2), not size.

**Step 4 — E3 as its own question, not a consequence of memory.**
- A lookup-like store will not produce relational transfer. Start from the narrow positives: the V9.10C shared NLMS head (77.5% never-taught rank pairs) and the ROLE code (+34 pp).
- Test whether role-bound codes (early/late traces), read by a shared error-corrective head, transfer **cyclic** relations. That head reached 51% in V9.10D.
- Include an engineered positive control and shortcut countermodels (E3-design skill).
- **Falsifier:** cyclic transfer stays at chance even with role codes, which would mean the representation must be learned (Hebbian/Oja on corroborated pairs, as in V45), not fixed.

**Step 5 — Process changes (no experiment, highest leverage).**
- **One reference learner:** the scaled sparse store plus the predictor from Steps 1–3. Every new idea is an arm against it on the same lifetime, with matched-budget and shuffled controls.
- **Scale before mechanism:** before adding machinery to fix a capacity failure, run the "just make it bigger" arm.
- **Make the resource contract follow measured requirements,** or state that it deliberately rules out the known solution.
- **Cap governance:** one design document per experiment, frozen; amendments go into the next experiment.

---

## 5. Confidence and gaps

- **High:** the recovered V39/V40 code reproduces its reports' directions (4 V39 runs and 1 V40 run). My prior non-reproduction was a bug in my own code.
- **Medium:** that capacity scales with KC units. The effect is large and monotone across 8 cells, but there is 1 seed per cell and no fitted law.
- **Low and untested:**
  - natural text;
  - storage efficiency after quantization and reduced per-byte writing;
  - any E3 path.
- **Provenance:** the V39 runner hash mismatch is noted in §0. V39–V44 runners were Claude-authored, and only the V41/V42 runs are recorded as Codex-executed. I did not examine V187 in depth (published today on the consolidated branch).
