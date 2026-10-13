# Cycle 2 — full technical histories and independent audit

All five arms completed one full reserved world with W, N_ALL and SHUFFLE: 5 jobs, 15 correlated histories, 7680 record applications. All independent audits accepted. 28 unit/tamper tests passed.

The implementation now logs and audits optimizer moments, step counters and observed-only EOS auxiliary targets. Probe restoration includes numeric parser and event metadata; checkpoints preserve all new state. The independent auditor rejects forbidden N writes, wrong shuffled targets, early auxiliary writes, decoder-target access and generated-target provenance.

The initial single-arm cost receipt is preserved separately. CPU, logging/digest work, worker RSS and serialization are separately reported. Scientific jobs: 0.
