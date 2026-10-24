# V145-R2 — revisions after the bounded exercise

The original V145 design remains prospective and unchanged. These are implementation/qualification revisions, not new arithmetic features or efficacy claims.

1. The combined trace wrapper must be a module-level pickleable object with explicit immutable receptor-table state. Test the whole learner + trace + RNG + cursor, not just the code learner. The repaired implementation is in `resume_audit.py`; the original failing attempt remains under `attempts/attempt1_checkpoint_failure/`.
2. Use an outer supervisor and append-and-flush attempt ledger before training. Save record/token counts and process CPU after every fit and on exceptions. Charge imports and child CPU. The present run's CPU is only a lower bound and cannot qualify total compute efficiency.
3. Check same-interface oracle *taught acquisition* alongside held accuracy. An oracle that cannot reliably acquire taught answers cannot support a strong negative about learned-code capacity. Keep its arithmetic priors outside candidate code.
4. Persist every frozen learned receptor table and its source/init/projection hashes before downstream fitting. This permits per-code nearest-neighbor controls, code-prediction/collapse diagnostics and restoration without replaying training. Missing tables in this exercise preclude those posthoc claims; do not recreate trajectories merely to retrofit them.
5. Require source/asset closure, exact exposure validation, resume/cursor checks and the complete selected development rehearsal before declaring a main dispatcher qualified. Current mutation tests cover only four result faults, not the entire proposed rejection matrix.
6. Keep the negative held result visible. Ordered pretraining has not beaten literal copy. No main launch, adoption, sample-efficiency claim or change of endpoint follows from technical completion. Gradual warmup, generic bytes, own EOS and the no-arithmetic-structure rule remain fixed.
