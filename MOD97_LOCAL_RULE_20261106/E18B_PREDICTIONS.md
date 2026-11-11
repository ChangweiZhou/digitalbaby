# E18b predictions (committed before the first E18b birth, 11 November 2026)

Brief: `OPUS_TASK_E18B_EVERY_NEXT_BYTE_20261110.md` with its four amendments; plan `E18B_E18C_E23_PLAN_20261111.md` (v3). Host: one 4-core cloud container, Python 3.13.18, torch 2.15.0+cpu; partners B2 and B3 rerun on this host (`_e18b`). Scorer: `e18b_e23_score.py`, written before any full birth.

"Hard mean" = mean of add89 and sub89 at k = 12, whole answers, final checkpoint, main head. "Max/min mean" likewise for max89 and min89. "Embedding movement" = `book_shift_by_role` at the last checkpoint.

- **P18b.1 (the input code is learned)** In B3-all the operator and the distractor-letter embeddings have each moved at least 0.2 of a row norm, in all three seeds. (B3 on file: 0.00.) Both roles must be present in the receipt.
- **P18b.2 (amended: the package learns something above shortcuts)** In at least two seeds, B3-all's max/min mean is at least 66 **and, in the same seed,** its hard mean is at least 10. Copy-operand shortcuts: 49–52 on max/min, 1–3 on add/sub. Held at about 40%.
- **P18b.3 (it stays far from the long trace)** B3-all's hard mean is at least 20 points below B2-all's in all three seeds.
- **P18b.4 (every-byte targets do not break a working package)** B2-all's hard mean is within 15 points of the same-host B2 partner's in at least two seeds. Held at about 50%.

Readings (from the brief, as narrowed by amendment 3): P18b.1 and P18b.2 holding → every-byte targets give the forward, within-step rule a path to its input code and the package learns; P18b.1 holding with P18b.2 missing → the input code was not the only blockage; this bounds the package at this dose and does not show every target needs longer credit; P18b.4 missing does not by itself show dilution. Births only: no life, retention or composition.
