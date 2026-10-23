# MiniFly v6 — learned procedure plus fast memory (design memo, 2026-10-23)

**Design only. Nothing built or run.** Sources were read as abstracts/summaries; caveats are stated per item. Claims about our own results come from the Codex branch reports (R2, external-memory, RECOVERABILITY) and earlier diagnosis in this repo.

## 1. Why the fly is still "stupid"
* As built, the fly is a fixed random expansion plus a learned linear vote: it answers with a similarity-weighted blend of taught answers. That stores facts; it cannot compose rules (v3: 100% taught / 0% held; R2: "1111" for everything).
* Rule learning needs (a) learned intermediate features, (b) reuse of one rule across positions, (c) working memory for intermediate state (carry). Project rules exclude attention/transformers/backprop and arithmetic-specific parts, so none of these come for free.
* Quantitatively (approximate): without a carry register, a lookup over a w-column window needs ~100^w patterns per output position and still errs ~5% (w=2) / ~0.5% (w=3) per digit, compounding with length. With reuse + carry state the rule is a 200-entry table (digit pair x carry).
* The 254-version mechanism ledger covers one axis only (importance, allocation, replay, retrieval, consolidation, metaplasticity, surprise, context gating, forgetting): how to store and protect associations. It has no entries for sequential state, reuse, learned gating, latent variables or working memory.
* Substrate defects (known, separate): low-dose writes behave mostly additively, ~10% address overlap spills between items, the end marker is biased negative, and NEW writes (~2.3x stronger per fresh warmup) overwrite OLD.

## 2. Constraints for v6
* Bitter lesson: structure must be learned from data in the long term. **Hard-wired step routing/pointers are ruled out.** Only generic primitives allowed.
* No attention/transformers, no backprop through time. Few-layer local credit (one-step into a state register) is allowed. Larger models allowed; blind scaling is not a fix.
* No arithmetic-specific structure; arithmetic is one probe of generic sequential computation. Toy non-arithmetic tasks first are allowed.

## 3. Human-brain mechanisms and how they map (read as abstracts/summaries)
| Finding | Bridge | Caveat |
|---|---|---|
| Arithmetic recycles older circuits (Dehaene & Cohen 2007) | generic circuits from broad data; arithmetic learned on top | summaries; critics prefer broader neural-reuse account |
| Counting gives way to retrieval with rising hippocampal activity/connectivity (Qin et al. 2014, PMC4286364) | slow procedure solves; memory stores confirmed results; retrieval takes over | Siegler waves from secondary summaries |
| Gating learned by dopamine prediction error in hierarchical corticostriatal loops (Frank & Badre 2012) | learned gating policy over working-memory slots replaces wired routing | computational model |
| Chunking: striatal start/stop signals bracket learned sequences (Martiros/Graybiel thesis) | repeated gate sequences compile into a reusable routine (learned loop) | interneuron role only proposed; least specified element |
| Competitive queuing (Averbeck 2002; human decoding PMC6436939) | existing decaying input trace is a recency gradient; winner-take-all + self-inhibition reads last-first, no pointer | human evidence non-invasive; gradient results unverified |
| Structure/content factorisation (Tolman-Eichenbaum machine) | content-free routine state, memory binds digits to it | model uses gradient descent; principle only |
| Sleep aids hidden-rule discovery (Wagner 2004; Darsaud 2011) | replay as rule extraction, only if representable | percentages secondary |
| Carries held in working memory; externalising intermediate results helps (Furst & Hitch 2000; Heathcote 1994; Vallee-Tourangeau 2016) | scratchpad via own-output feedback | mixed evidence |

Other computational candidates found: AuGMEnT (synaptic tags, local RL working memory; PMC4351255), latent-cause/clone-structured models (Gershman et al. 2010; CSCG PMC8062558; Kappel 2014 PMC3967926; HTM arXiv 1511.00083) as fallback for discovering hidden state, e-prop (PMC7367848), evolved Hebbian rules (arXiv 2007.02686; no algorithmic-task precedent found), genomic bottleneck (PMC11420173). No prior work found that learns multi-digit arithmetic from raw bytes with local rules.

## 4. Proposed design
Generic primitives only; everything else learned.
1. Working-memory slots with learned input/output gates, trained by a dopamine-like error with eligibility traces.
2. Content-free state register with one-step local credit (approved).
3. Chunking: successful gate sequences become routines (start/stop bracketing).
4. Procedure to memory: confirmed procedure outputs are written into the mushroom body; arbitration by speed/accuracy history; retrieval takes over for frequent items.
5. Sleep replay for rule extraction, tested against waking practice.
6. Optional generic motif: competitive queuing over the existing recency trace (needs approval).
Data option: worked-example text as ordinary data; run with it first, then ablate to measure what the fly finds alone.

## 5. Ladder (non-arithmetic first; each with a control that must fail)
1. Learned gating: hold a cue across distractors (1-2-AX style) from reward only; control: always-overwrite fails.
2. Serial recall of random byte lists, forward and reversed, unseen lengths.
3. State: running parity / sum mod 10 with output at end; stateless readout stays near chance.
4. Two lists in lockstep, then addition with carry; compare with memory-only baseline; count samples. Per-digit accuracy separates cases: ~55% no state, ~95% two-column window, ~100% true state.
5. Human signatures: ticks per problem fall as memory takes over; replay beats waking practice on unseen items.
Prediction to falsify: with reuse + carry, ~10^3–10^4 problems suffice versus ~10^6–10^7 for windowed lookup.

## 6. Open decisions
1. Competitive queuing allowed as generic motif, or serial order also learned via gating?
2. Worked-example text allowed in the training stream?
3. Larger compute budget and fast harness approved (learned gating needs far more trials than past 15–30 min CPU-capped runs)?

Not recommended: more protection gates, more teaching strength, larger cell layers, more replay.
