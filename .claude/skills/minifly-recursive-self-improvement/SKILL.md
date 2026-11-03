---
name: minifly-recursive-self-improvement
description: "Use for explicitly requested MiniFly recursive self-improvement of experiment selection and verification, not a new learner design. Diagnose process incidents, implement executable general checks, run contrasting regressions, and seek independent challenge."
---

# MiniFly recursive self-improvement

This is a **repository-local** Claude/Codex skill, not an installed ChatGPT plugin skill. It is DISTINCT from minifly-research-loop, which designs model experiments. The old MRK-1 command dispatcher is missing; do not claim formal checks, seals or independent approval.

## Procedure

1. Determine whether the user asked for recursive self-improvement of the **research process**. If so, do NOT allocate or train another MiniFly model. Identify a real process incident, the preventable decision and the measured CPU/record/memory cost.
2. Propose a generalizable repair and at least one should-fail and one should-pass example. A check fitted to the previous incident's exact words is insufficient.
3. Implement a machine check when feasible. Actually run: python3 -m unittest discover -s tools/minifly_meta_guard -p 'test_*.py' -v. Record failures and passes. Do not invent an MFC PASS.
4. For an explicitly meta-only change, run: python3 tools/minifly_meta_guard/guard.py verify-scope --mode meta --base-ref BASE --head-ref HEAD. Model experiment directories and the version registry must not change in that scope.
5. Before constructing an oracle or solver, run: python3 tools/minifly_meta_guard/guard.py admit-solver --plan PLAN.json --source-file SOLVER.py. Success is merely conditional technical admission, not a proof of available RAM or learner performance. Execute technical trials using run-capped with a memory and time budget; use OS/container hard isolation where required.
6. Commit process changes on a separate governance branch. Request another agent family to produce genuinely novel counterexamples before protected changes are merged. Do not self-approve.
7. Judge future self-improvement by **unseen incidents prevented without false blocks**, not how many tests, memos or versions were generated.

Known case: V185's multiclass LP witness timed out without an 80/80 representation certificate after RSS reached about 3.8 GB versus 1 GiB. The guard has executable scope, memory-arithmetic, source-pattern and runtime tests. They remain partial, can have false positives, and are not the missing MRK-1 verifier.

See the original minifly-governance skill for protected-path/independent-review policies. Tests are at tools/minifly_meta_guard/test_guard.py. Review status: PENDING.
