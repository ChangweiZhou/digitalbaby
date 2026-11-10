# 9.62 — E16-F7: six more drift seeds; the signature S separates failures but does not rank

E16-F7 ran six new births (1100009, 1200007, 1300021, 1400017, 1500007, 1600033), each with a drift learner (raw-byte reservoir of 65,536 slots, admitted online) and a drift twin. The early signature S was pre-registered this time: the learner's lowest mod-97 mean at k = 12 over checkpoints 2–4.

- **Births, twins and the rate hold (P16F7.1–3).**
  - The births reach 86.3–100 at k = 12.
  - Every twin ends at or above 93.7 / 91.2 / 92.2.
  - All six learners pass the family gate. 1600033 passes by 0.005 on the mod-97 operations.
- **The signature misses (P16F7.4).**
  - **Thresholds hold:** the two seeds below 65 fail on compositions, and the twelve at or above 70 pass. But both seeds below 65 were the ones S was chosen on; no new seed fell below 65.
  - **Correlation misses:** over eighteen seeds the Spearman correlation between S and the final `+-` is 0.109, against the 0.35 required (0.399 on the twelve seeds on file).
  - **Why:** the new seeds rank almost in reverse. 1400017 has S 65.2 and the best final `+-` of all eighteen (79.0). 1200007 and 1300021 have S near 82 and lose 15 and 30 points of `+-` during the mod-73 blocks, after the window S looks at.
  - So, as the brief's reading put it beforehand, S separates the known failures but does not rank the passing seeds.
- **The hard-pair gate keeps disagreeing with the family gate (P16F7.5).**
  - Of the new seeds, 1 of 6 reaches 0.80 of the twin on `+-`/`-+`, while 6 of 6 pass the family gate.
  - 1200007 passes the family gate with `+-` at 49.7, about the failing 400009's 48.0.
  - Over eighteen drift seeds: 16 pass as pre-registered, 4 pass on the hard pairs alone.

**Run.** 79,218 CPU-s, 5 h 37 min of wall time, on a 2.10 GHz cloud Xeon, at harness commit e78aa83e. There were no restarts or reruns, and the forecast was 4% low on CPU.

Development result on a toy harness, evidence level E2, one stream per seed. The "local" status of the sign rule is still open (9.60), and no learning-core improvement is claimed.

Sources: `E16F7_REPORT.md`, `E16F7_PREDICTIONS.md`, `exp45/SUMMARY_E16F7.md`, `exp45/SIGNATURE_E16F7.md`, the twelve receipts, the final weights and the six births.
