# MiniFly qualification for the three remaining protocols

**The three engineering cycles completed for all three prototypes.** The short rehearsals exercised learning, scoring and fresh-process restart. The locked behavioral doses and admission gates are still ahead.

## Cycle 1 Fixtures and controls

- **Frozen encoder.** The A and B fixtures each had 512 feedback prefixes and 64 held presentation keys. The A and B actor bytes were disjoint and used the same opcode vocabulary. The consistent and inconsistent controls used the same chronological query prefixes. Their outcomes differed on exactly one quarter of the presentations across two complete epochs, as intended when half the semantic keys have law-dependent outcomes.
- **Downscaling.** The split contained 4,704 taught prompts, 2,351 monitoring prompts and 2,354 sealed prompts. Every one of the 9,409 ordered mod-97 prompts passed the answer oracle. Both orders of a pair stayed in the same partition. The decimal carry fraction was 42.5–43.4% across partitions.
- **Attractor.** The counting curriculum exposed 75 ordered sums and withheld 25. No held equation appeared as a teaching prompt. Traces contained tally and padding bytes, with no decimal result. The shuffled and aligned conditions matched trace-target counts. Count labels intentionally exposed all nine possible held result values, 10–18. This is supervised counting experience and is part of the evidence boundary.

[Fixture receipt](CYCLE1.json)

## Cycle 2 Intervention paths

- **Frozen encoder.** The online output-error rule changed the local readout and preserved the encoder's tensor hash. The stock and recurrent routes produced a causal feature row for each target site.
- **Downscaling.** One pulse and 128 smooth multiplications reached the same nominal factor within float32 tolerance in a zero-update check. Only the shared weight matrices and byte embedding shrank. The biases and fixed-sensor readout retained their values.
- **Attractor.** Both models began with identical trainable tensors and degree-four connectivity. The ring had 32 triangles and the rewired graph had five, confirming different topology. The zero-input test found eight distinct stable localized states in the ring and seven in the rewired control. This qualifies state persistence, with learned tally-driven displacement and a causal role in arithmetic still open. Changing auxiliary trace targets changed the worked loss and left the answer-only masked loss unchanged.

[Intervention receipt](CYCLE2.json)

## Cycle 3 Short training and restart

| Protocol | Rehearsal dose | Native observations |
| --- | --- | --- |
| Frozen encoder | 4,096 developmental targets per trained birth and 16 local histories with 64 A and 32 B episodes each | All B probes scored 0/8 exact at both checkpoints. The encoder tensors stayed frozen. The planned forward-transfer endpoint needs the full acquisition and B-dose checks. |
| Downscaling | 16 common updates and 128 updates per branch at batch 64 | All three arms scored 12/12 on a small exposed sample and 0/12 on sampled held sums. The engineering fork occurred at a fixed step to exercise the controls. The main experiment's acquisition plateau was outside this rehearsal. |
| Attractor | Six arms with 75 count, addition and auxiliary groups each | All arms scored 1/13 sampled taught sums and 0/25 held sums. The ring retained eight localized supports after training. The rewired control retained six. |

For each protocol, a fresh process reloaded one checkpoint and reproduced the next update's model or readout hash exactly. For the two gradient references, the restored checkpoint included optimizer state. These tests support the implementation paths. They provide no behavioral evidence of forward transfer, a late rise on held arithmetic, or attractor-based counting.

[Training and restart receipt](CYCLE3.json) · [Three experiment designs](../minifly-whole-repo-audit/MINIFLY_THREE_REMAINING_EXPERIMENTS_DESIGN_20261106.md)

## Reproduction

Run from the project root in the PyTorch environment. Set `archive` to the local archive with the V216 and V181 source files. The encoder fixture also uses the cached V207 text file `output/minifly-pretraining/books/2600.txt`.

```sh
runner=output/minifly-remaining-experiments/qualify_remaining.py
scratch=$(mktemp -d)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_USE_SHM=0
python -B "$runner" 1
python -B "$runner" 2 "$archive"
python -B "$runner" 3 "$archive" "$scratch"
python -B "$runner" resume "$archive" "$scratch"
```
