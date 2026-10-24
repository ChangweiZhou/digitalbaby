# Cycle 1 — design, audit, fixtures, revision

This is author-run engineering qualification, not independent review or formal science.

The published V144 design and historical V143 science are preserved. The implementation is a new instrumentation runner plus a separate evaluator reference. The native runner calls the original V143 update exactly once. The reference has generic FE-PERM context, a learned ReLU layer, nine output bits, and its own EOS; no operand parsing, arithmetic features or restricted answer vocabulary.

## Tests and corrections actually made

1. Restored missing published source and birth assets by SHA256 from the existing source archive and V143 Git objects. The first import found missing configuration; another constructor exposed a lazy-import dependency omitted by the old runtime-closure list. Source/assets, not experiment traces, were restored. No old learning history was rerun.
2. The first fixture process hit a Python namespace collision: our original `common.py` shadowed the inherited native module. Renamed it `v144_audit.py`.
3. The second attempt failed exact FE-PERM comparison: using literal `.95` instead of the inherited time/decay API changed low bits. Replaced the approximation with the actual parent front-end API. Prompt and generated-prefix tests now pass.
4. Audit found the first finite-difference fixture selected inactive units and therefore tested only zero gradients. Revised it to select nonzero gradient entries for W, b and U. Maximum error in the final qualified run: approximately 3.92e-11.
5. Three successful native instrumentation fixture executions each compared two learners over eight taught records. Full state equality held after every record. The qualification also checks nonmutating reads, all 257 observed tokens, invalid 511 remaining invalid, arbitrary-byte payload/EOS serialization, first-record zero warmup and nested orbit separation.
6. Fail-closed mutation fixtures reject missing/extra/duplicate histories, closure failure, wrong hashes, missing metrics, incomplete histories, a one-world substitution and invalid settings. Command-line exit tests follow in cycle 3.

The final passing fixture output is `CYCLE1_CHECKS.json`; `CYCLE1_SOURCE_LOCK.json` includes runtime imports after an actual native birth, required assets and the bounded coordinator before Stage B begins. Failed logs remain in this directory. The short setup commands before the ledger existed have disclosed CPU-accounting gaps; zero native record applications occurred in those failed imports.

## Authorization and scope

The user's three-cycle request authorizes these fixtures, the designed paired development diagnostic and a short addition interface trial. It is not treated as authorization for the six-CPU-hour full Stage C grid or its H1/H2 holdbacks. No formal native addition campaign is included.

Stage B runs LEARN/FROZEN on the original development D2 only, 150 passes, with at most two single-thread workers and caffeinate following the coordinator. Sources and learning settings are fixed before those two histories. No failed history is automatically restarted.

All technical exposures are reported separately from the paired diagnostic. In this cycle: 48 native fixture record applications and 12 synthetic reference record applications (36 observed target tokens). Attempts that failed before training remain listed, not silently discarded.
