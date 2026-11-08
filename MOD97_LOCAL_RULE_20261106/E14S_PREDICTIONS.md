# E14S prospective predictions

Source brief: CODEX_TASK_E14S_DECAY_CONTROLS_20261108.md at beaad27c, consolidated branch. User authorized implementing and running the assigned part. No E16 process or source is changed.

Fresh seeds 500011, 600013, 700027 passed the prelaunch collision scan in E14S_SEED_CHECK.json. Standing seeds are 73129, 93683, 116089. No substitution based on scores.

1. A: BP arm D at decay 1.0 has final old-operation mean <=85% in every standing seed. >=92% would support a remaining decay asymmetry; intermediate outcomes are inconclusive for that dichotomy. Compare with existing D0.3, C1.0, C0.25. These arms also differ in replay/teaching, so this alone does not isolate a mathematical property of the update rule.
2. B: at most one new seed has any held pair below50% final-answer accuracy. At least seven of nine local0.25 seeds (E13 standing, E14 fresh, E14S fresh) have held composition mean>=75%. Report every pair and complete scratchpad score. New BP references stay at0.3.
3. C: seed400009 `+-` never exceeds40% on logged composition checkpoints; `-+` exceeds60% by update2048. Log at1024,2048,...8192. Logs must leave model, optimizer, RNG and replay unchanged. Full seed400009 scientific receipt must exactly match E14 apart from cpu and the additional compose_checkpoints key. Default behavior must reproduce the frozen runner.

The command roster and environment are frozen in E14S_LOCK.json before launch. Every command uses the qualified e14_acceleration_worker.py inference wrapper, including births. Births use the existing development defaults; lives use batch64, blocked schedule, half replay, uniform buffer1024, lr0.001. A uses D/wd1; B uses C/wd0.25/log1024 and D/wd0.3; C reruns C400009/wd0.25/log1024. Thirteen scientific subprocesses:3A+3birth+6B+1C. Short technical parity fixtures are additional, not scientific results. No tuning, new panels or new mechanisms. Validate only the exact completed receipt; a technical/parity failure stops the affected part. If C fails parity, stop its dependent logged B candidates too.

E14 recorded CPU lower bounds: six births3903s,21E8 lives26702s. Scaling those averages gives approximately4CPU-hours for3births+10lives, with overhead and additional scoring; the brief's7CPU-hour allowance remains prudent. Six E16 workers currently each consume about one core. Six additional single-thread workers may contend with them: budget roughly2-3 wall-hours including verification, not a promise of1.5hours. Actual per-job CPU and wall will replace this forecast.

The full logged seed400009 rerun gates new logged B lives. Independent A lives and births may run alongside it. Successful completion requires E14S_REPORT.md, raw receipts, all per-seed tables, costs, and review section9.28. No automatic push was requested for E14S.
