**9.72 Literature check on local credit through time and layer-local targets (cloud session, 10 November 2026; `PROJECT_DOCUMENTS/LIT_CHECK_20261110/LITERATURE_CHECK_LOCAL_CREDIT_IN_TIME.md`).**

**Scope and method.** This was a read-only job. About fifty primary sources were opened in the session, and eight central citations were checked against the text by hand.

**The four claims that steer E17 Phase 4 and E18.**

1. **Within-layer credit: confirmed. Cross-layer credit: corrected.**
   - The exact online gradient within a diagonal-recurrent layer costs about one trace per parameter.
   - The main multi-layer methods (Zucchet 2023, OSTL) drop the credit that passes through an upper layer's memory. Zucchet's LRU training is still within 1–4 points of BPTT on Long Range Arena at about 1M parameters.
   - K-prop (NeurIPS 2026) carries that credit with one extra trace per synapse per distinct downstream time constant, exactly for two leaky layers. That is the closest published case to a two-bank learner with fixed retention.
2. **Block-local objectives: corrected.**
   - With few blocks they match backprop even on ImageNet; they fall behind as the number of local pieces grows.
   - At language-model scale, one workshop study at 125M and 400M parameters exists. It never matches end-to-end training, and its per-block next-token heads show "representation myopia": the blocks above the first barely improve on what they inherit. That is the E18 local-heads design.
3. **Sign symmetry: corrected.**
   - With a per-layer scale or normalisation it comes within about 4–11 top-1 points of backprop on ImageNet ResNets. Unscaled, it can diverge.
   - FA and DFA collapse on ImageNet. A controlled depth effect is not shown.
   - **No study of sign symmetry through time or in recurrent networks was found.** E17 Phase 4 is therefore outside the published evidence.
4. **Fly dopamine: confirmed, narrowed.**
   - Dopamine timing sets the sign of KC→MBON plasticity within about ±1 s, in particular compartments.
   - Dopamine neurons also carry movement, state, novelty and forgetting signals.
   - Adult plasticity is reported at every stage after the receptor neurons; the receptor-neuron layer is comparatively stable after an early critical period.

**Suggestions, not decisions.** The memo offers three: K-prop-style traces through the upper bank's fixed retention; a per-layer scale on sign feedback; and for local heads, a look-ahead or upper-layer-informed auxiliary in the lower bank's head.

**What it changes.** Claims 1–3 should be quoted in their corrected form from now on. The memo decides nothing.
