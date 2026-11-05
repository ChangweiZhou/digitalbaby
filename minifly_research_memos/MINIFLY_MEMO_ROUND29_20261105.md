# Memo: round 29 — evolution with a content-addressed memory in the search space (2026-11-05)

Evidence: dev. One 40-generation ES run (16 genes, about 2,600 lifetimes), plus three small expressivity probes. Same people-only MiniWorld lives and validation as round 28. Code and logs: `minifly_review_experiments/round29_evolved_memory_20261105/`.

## What was added to the search space (`inner4.py`)

A generic episodic store:
- **Addresses:** hashed byte n-grams (orders 2–4) of the last W bytes, weighted by order and by distance decay τ. There are no words and no segmentation.
- **Writes:** at every byte, with replacement fraction ρ, scaled by the surprise modulator.
- **Reads:** a softmax with gain β, mixed into the prediction with weight α.

W, τ, the order weights, ρ, β, α and the modulator are genes.

## Results

| arm | val bpb | answer bits |
|---|---|---|
| GLN reference (round 19) | 0.64 | 5.74 |
| evolved, memory available (gen 40) | 1.23 | 9.55 |
| same genome, memory off | 1.19 | — |

Over the ES run, the memory mixing gene went from 0 to −3.2: weight about 4%. **Evolution turned the memory off.**

**Expressivity probes** (hand-chosen points, used only to test the search space, not to tune):

| probe | finding |
|---|---|
| fixed mixing α, any W / β | memory always hurts (answer bits 15–26 vs 10.8 without memory) |
| α learned online by a local gate (`inner5.py`) | the gate learns to ignore memory: no harm, no help |
| + n-gram habituation / IDF (`inner6.py`) | no help |
| + write only when surprised (`g0` < 0, `g1` > 0) | memory top-1 at the answer's first byte 0.07–0.23; p(answer) from memory ≈ 0.01 |

**Diagnosis.** Writing every byte under recent-n-gram keys makes the store a local "what comes next" model. The rows keyed by "Goda" are overwritten by whatever follows "Goda" (" went", "?", "\n"), not by the place. The binding stores that worked earlier had two ingredients that this parametrization cannot express well:
- *value-onset writing*: write the value when it starts, not every byte;
- *salient-key selection*: key on the entity, not the most recent bytes.

ES started from a point where memory hurts, so it could only move downhill, toward turning memory off. That is an exploration and expressivity limit, not proof that memory is useless.

## What this says, in Bitter-Lesson terms

**Two runs (rounds 28–29) agree:**
- Search over a small genome of a generic substrate reliably buys **generic prediction** (bpb 4.0 → about 1.0).
- It does **not** discover binding.
- The ingredients that make binding work have been, in every success so far, either hand-supplied (rounds 10–26) or provided by a learned gating mixer (round 20 GLN + recall: answer bits about 5 → 3.2).

**The most scalable working substrate we have is GLN context mixing plus recall experts.**
- Its gating learns *when* memory is trustworthy, locally and online.
- That is the role the fixed-α / single-gate versions here could not fill.

## Next iteration (proposed)

1. **Make the GLN + recall-expert learner (round 20) the inner substrate.** Evolve its *structural* genes by ES, with fitness = answer bits on fresh worlds:
   - recall-key types and orders;
   - decay;
   - hop count;
   - write gating by surprise;
   - expert count;
   - mixer gating contexts.
2. **Seed ES from a population of diverse starting points** (not one μ), so memory-on regions can be found from the start rather than only reached downhill.
3. **Scale test:** the same genome at 2× / 4× contexts and recall capacity. Does the evolved learner improve with size?

**Stop rule:** if ES on the GLN substrate also cannot push answer bits below round 20's 3.2, the conclusion is that binding must currently come from a structural prior, and I will say so plainly.
