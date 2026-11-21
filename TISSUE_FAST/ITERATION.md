# How a round of experiments is run from now on (21 November 2026)

Owner, 21 November 2026: "实验要短，快速迭代。"

- **A round takes at most about 2 minutes of wall time.** Default: one seed, up to 4 settings (one a core), **4,096 bytes, one reading window of the last 2,048 bytes** (`evt_run2.py --bytes 4096 --window 2048`), the second run without records off. Readings: `tab.py` (S(d,0) at every distance) and `letters_read.py` (the same on positions where no earlier byte predicts the present one, with a shuffle level).
- **Calibration of the short form** (this commit; runs `Q_*`): 4 tissues of 4,096 and 2,048 bytes ran in 12 s of wall time, their readings in 20 s. At 4,096 bytes both of plan 8 and 9's best settings give the same farthest distance as their 16,384-byte runs, by both readings (first form, seed 73129: 8 by S(d,0) and 11 by the letter reading, against 8 and 11; coincidence form, seed 93683: 11 and 11, against 11 and 11). At 2,048 bytes the first form is not formed yet (6 and 9).
- **Longer lives, more seeds and both forms** only to confirm a setting that has won a short round, in one batch.
- A plan is a few lines: the question, what refutes it, the settings. A result is reported as found only after the letter reading.
