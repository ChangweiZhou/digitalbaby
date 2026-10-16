# Cycle 2 — checkpoint and functional-diagnostic audit

Design remains the same unseen-addition HELD/global-sample comparison. This cycle tests state/control infrastructure, not accuracy thresholds.

Required fixes/checks before full run:

- Snapshot the whole recurrent learner and native context, not just plastic arrays or a digest. Store arrays by content with independent object tokens; same-content mutable arrays must remain separate after load.
- Bind cursor, next record ID, model variant and manifest. Wrong variant, absent q/p, corrupted payload/assets, and changed cursor must be rejected.
- Native mid-answer q/p restoration must preserve next-byte logits. Preserve raw bytes including NUL/newline/space/0xFF without semantic preprocessing.
- Offline ALL means exact copied post parameters (not floating-point readdition of a rounded difference); moments remain pre because these are read-only functional interventions.
- Independent auditors must reject permission tampering, actual state-write evidence tampering and incorrect own-output grades.

Budget plan: 15 technical record applications; restores and seven-view queries create no new training trajectories. Any failed attempt remains charged in the durable ledger.
