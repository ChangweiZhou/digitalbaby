# Next-core mechanism implementation: qualification build

This directory implements the 6 October registered Q/P/S/REL design in SPEC_LOCK.md. It imports unchanged frozen Full151, R_center, lifetime/reuse and Latin sources. The scientific parameters and source contracts are not selected from development outcomes.

Current entry points are technical qualification only:

```sh
/private/tmp/minifly_next_core_20261006_env/bin/python qualify.py --cycle 1
/private/tmp/minifly_next_core_20261006_env/bin/python qualify.py --cycle 2
/private/tmp/minifly_next_core_20261006_env/bin/python qualify.py --cycle 3
/private/tmp/minifly_next_core_20261006_env/bin/python qualify.py --freeze
```

Cycle 3 is relatively expensive: it runs complete development jobs for every registered arm and assay, actual SIGTERM/checkpoint recovery tests, and four/eight-process parity checks. Receipts are committed atomically; a committed development job is skipped. Do not rerun merely to view progress.

The worker rejects official world IDs. This build has no official-science dispatcher or final-profile analysis driver. Qualification success would establish E0 implementation evidence, not mechanism efficacy, E1 retention gains or E3 transfer. Before an official launch, the registered exclusion registry, roster, dispatch, analysis and total resource acceptance must be complete. No science is authorized by a PASS record.

Dependencies: requirements.txt. Python is pinned to 3.11.5. SOURCE_LOCK.json is generated only when all three qualification records match the same source identity. Each operative Full151 store is independently born through the original canonical constructor.

Read results/THREE_CYCLE_REPORT.md for the completed qualification outcome, or results/cycle3.log during development execution.
