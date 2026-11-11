**9.74 Literature check on learning and memory at several timescales (cloud worker session, 11 November 2026; `PROJECT_DOCUMENTS/LIT_CHECK_20261110/LITERATURE_CHECK_MULTI_TIMESCALE.md`).** This was a read-only job. About 25 primary sources were opened in the session, and the central numbers were checked against their raw text. It answers Mr. Zhou's question in three parts.

- **The Hope paper is reading C, not oscillation.**
  - Its "frequencies" are how often blocks of MLP weights are updated (Nested Learning, Sec. 7.1, Eqs. 70–71).
  - Everything is trained by backprop (AdamW).
  - The brain-wave hierarchy is cited as inspiration for ordering the levels; it is not tested.
  - Its continual-learning gains appear only in a figure, on 3B–8B Llama backbones, against ICL, EWC and InCA. There is no replay baseline, no seed count, and no independent replication was found.
- **Brain rhythms do not hold information for 120 s.** They run at about 0.5–150 Hz, and cortical intrinsic timescales are about 50–350 ms. Over seconds to minutes, the documented carriers are replay, ongoing neuromodulator activity and synaptic processes at several rates. In the fly, slow calcium oscillations in the MP1/MV1 dopamine neurons, about 0.05–0.3 Hz, gate long-term memory after spaced training (Plaçais et al. 2012).
- **The five claims.**
  1. Paper: confirmed in outline, corrected in detail.
  2. Reading A: confirmed. The choice of rates matters, and in one controlled ablation (S4D) real decays lost 5–10 points to complex ones.
  3. Reading C: confirmed, narrowed.
     - Benna–Fusi chains (Kaplanis et al. 2018, 2 RL tasks, 3 runs) and a metaplastic hidden weight (Laborieux et al., six permuted MNIST tasks, about 97% against 8–13%, matching EWC without task boundaries) buy retention without rehearsal. Both are synapse-local and need no importance estimate.
     - Nothing was tested on sequences, against a replay-share curve, or with sign-based credit.
  4. Cost of non-element-wise recurrence: confirmed for forward-only credit. Under the backward-sweep rule of the 9.82 addendum, an inner loop costs K times the work per byte instead.
  5. Fly parallel memories: confirmed (9.72). Oscillation's role: narrowed. Theta–gamma ordering is a supported hypothesis, and no function is shown for the fly's 10 Hz mushroom-body oscillation.

Looped models match deeper ones at equal compute (Saunshi et al. 2025), so what an inner loop buys is parameters. No controlled test was found of oscillating units against plain decays for holding ordered items.

**Suggestions, not decisions.**
- A three-variable Benna–Fusi chain per synapse in the window learner, tested at replay shares 0–0.5 against learning-rate-decay and random-consolidation controls. Cost: 2 floats per synapse.
- A shared 512×512 loop run K times per byte in the two-bank learner, against unshared layers at matched compute. Cost: about 262K weights and K× work per byte.
- Complex retentions only if a failure is shown to be about order.

The memo decides nothing.
