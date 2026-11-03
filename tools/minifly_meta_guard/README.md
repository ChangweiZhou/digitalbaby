# MiniFly partial recursive meta-guard

This **runs**, but it is not MRK-1 or an installed ChatGPT skill. It does not certify science.

Run executable tests:
    python3 -m unittest discover -s tools/minifly_meta_guard -p 'test_*.py' -v

For the user's explicitly typed **meta** workflow, reject accidental scientific model or registry edits:
    python3 tools/minifly_meta_guard/guard.py verify-scope --mode meta --base-ref BASE_SHA --head-ref HEAD_SHA

For actual numeric/oracle solver source, estimate memory before building arrays:
    python3 tools/minifly_meta_guard/guard.py admit-solver --plan tools/minifly_meta_guard/example_bad_v185_solver.json --source-file SOLVER.py

Run an admitted technical command with monitored resources:
    python3 tools/minifly_meta_guard/guard.py run-capped --rss-mib 1024 --wall-seconds 90 -- python3 solver.py

Limitations: the agent must establish the **meta** intent from the request; this tool does not parse a chat. Sparse estimates are supplied rather than independently proven. Runtime checks poll RSS of a POSIX process group, which is NOT a hard OS RAM cap. Optional --hard-as limits virtual address space and can reject valid mapped libraries; use container/OS limits for strict isolation. Source heuristics may flag a safe sparse calculation and require a bounded-workspace proof. The full MRK-1 CLI, blinded canaries, cross-agent verification, and integration into the default branch are absent.

This is a governance-review proposal, not a changed model version.
