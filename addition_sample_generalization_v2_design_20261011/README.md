# MiniFly SAMPLE-GEN V2 design

Primary question: how much persistent ability on genuinely unseen combinations
does the core acquire per external training sample?

- [EXPERIMENT.md](EXPERIMENT.md): complete prospective protocol and nine configurations.
- [BENCHMARK_CONTRACT.json](BENCHMARK_CONTRACT.json): machine-readable benchmark,
  budgets, roster and admission rules.
- [DESIGN_REVIEW.md](DESIGN_REVIEW.md): completed task review and remaining gates.
- [BENCHMARK_FIXTURES.json](BENCHMARK_FIXTURES.json): passed pure task/exposure checks.
- [benchmark_fixtures.py](benchmark_fixtures.py): reproducible standard-library checks.

```bash
python3 benchmark_fixtures.py
```

This command verifies task geometry, clean solvability witnesses, synthetic
exposure and budget accounting. It imports no native core and runs no scientific
learner. It does not qualify the mechanisms or replace the three implementation
qualification cycles.

The recommended unrun WG1/SF1 efficiency proposal is superseded prospectively.
Historical experiments and their accepted results remain unchanged. No experiment
has been launched or new design published to GitHub.
