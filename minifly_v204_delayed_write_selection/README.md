# MiniFly V204 — Delayed Write Selection

All 75 official runs completed. The delayed gate passed the canonical guardrail and failed the promotion gate. It missed every destination start in both cue-change conditions and triggered on every distractor. Correct-position accuracy was 99.6% with the familiar cue missing, 99.9% with the misleading cue, and 16.5% with a distractor before the destination.

The bundle contains the engineering checks, sealed story events, frozen plan and per-run emitted bytes. The test used the frozen QA1 protocol without gold-answer feedback.

## Reproduction

Use Python 3 with NumPy and SciPy. From the repository root, create the local aliases expected by the source-locked runners:

```sh
ln -s minifly_v202_babi_qa1 minifly-v202
ln -s minifly_v203_babi_qa1_binary_gate minifly-v203
ln -s minifly_qa1_frozen_generalization minifly-qa-generalization
cd minifly_v204_delayed_write_selection
OPENBLAS_NUM_THREADS=1 python -B qualify.py
OPENBLAS_NUM_THREADS=1 python -B official.py
```
