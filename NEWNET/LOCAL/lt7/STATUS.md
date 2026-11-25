# lt7 status (25 Nov 2026)

`lt7` is a local tissue built on the template of the cerebral cortex. Each component, its biological source and its implementation are listed in the header of `lt7.c`.

## History

| Version | What it was | Status |
|---|---|---|
| `lt5` | first design from the owner's definition of a cell | kept for the record |
| `lt6` | cerebellum-based design | abandoned: the goal is the cerebrum, not the cerebellum |

`lt7` adds these changes:
- tag-and-capture consolidation;
- a metabolic budget for each part of a cell, so connection counts emerge from competition for resources;
- distal input that primes a cell only when it coincides with basal input;
- a passive synaptic decay 10 times slower than before.

## Checks (`checks/checks.py`, size 2048/512/512)

| Check | Result |
|---|---|
| test leaves no trace | pass |
| requirements audit | pass: fan-in emerges (mean 4.8, range 1–10); no silent cells; 0 locality violations; only feedback cells read the answer |
| 7-byte cycle | pass (0.03 nats/byte) |
| learned connections reach the output | pass: English 3.48 vs 3.80 with biases only, at 64k bytes |
| cost | per-byte time grows with cells at exponent 0.98. A 160k-byte life at 2× size is flat from 48k bytes on (about 300k synapses, about 2.3 ms/byte). The check's late/early ratio is 1.58, because its early window falls inside the development phase |
| memory 41 bytes back | **fail** (about 5 nats/byte) |
| relay helps | **fail** (no difference) |

English loss on 5 held-out windows falls through the life: 3.73, 3.58, 3.50 and 3.48 at 16k, 32k, 48k and 64k bytes.

## Open points

- Early learning is slower than with the fast decay. The earlier version reached 3.24 at 32k bytes.
- Credit from the output end does not yet reach the tissue usefully.
- The tissue does not yet hold information 40 or more bytes back.

The literature audit of the first biology table is `../lt5/AUDIT_BIOLOGY_TABLE_20261125.md`. The English and addition data are in `../arith/`.
