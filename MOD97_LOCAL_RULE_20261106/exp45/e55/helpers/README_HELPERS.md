# The helpers that launched E55's checks and runs

These shell and Python files are copied here unchanged, as they were run. They lived outside the repository, in `/home/user/logs51/` on the worker's machine. No receipt is changed by adding them.

| File | What it did |
|---|---|
| `e55_checks.sh` | The first launcher of checks 1 and 2 and of the four dry runs of check 3. Checks 1 and 2 ran from it and wrote their receipts. Its four dry-run lines failed at once on a quoting error in the shell, before any update, and wrote only an error line each. Their logs are `logs/aborted/dry_unrelated_S.1795011550.log`, `dry_unrelated_B.1795011550.log`, `dry_conflict_S.1795011550.log` and `dry_conflict_B.1795011550.log`. The launcher was then stopped. |
| `e55_dry.sh` | The second launcher. It ran the four dry runs, starting each only while fewer than three Python processes were running, then waited for checks 1 and 2 to end and ran `python3 exp55_reserve.py check 3 73129`. Check 3 found the four dry receipts and read them. |
| `e55_dry.py` | Called by `e55_dry.sh` as `python3 e55_dry.py <task> <rule>` (with the virtual environment's Python). It imports `exp55_reserve` and calls `run(task, "blind", rule, 73129, OUT55 / "checks" / "dry", <description>, updates=64)`. Because the call goes through `run` and not the command line, each dry receipt records this description in place of a command. |
| `e55_runs.sh` | The launcher of the eight runs: `e55_runs.sh blind`, then `e55_runs.sh shut`, three processes at a time, each running `python3 exp55_reserve.py run <task> <arm> <rule> 73129`. |

The command behind each dry run of check 3:

| Dry receipt in `checks/dry/` | Command run (from `MOD97_LOCAL_RULE_20261106/`) | Description recorded in the receipt |
|---|---|---|
| `e55_unrelated_blind_S_73129_dry64.json` | `python e55_dry.py unrelated S` | `python3 exp55_reserve.py check 3 73129 (its dry run unrelated S, started by a helper)` |
| `e55_conflict_blind_S_73129_dry64.json` | `python e55_dry.py conflict S` | `... (its dry run conflict S, started by a helper)` |
| `e55_unrelated_blind_B_73129_dry64.json` | `python e55_dry.py unrelated B` | `... (its dry run unrelated B, started by a helper)` |
| `e55_conflict_blind_B_73129_dry64.json` | `python e55_dry.py conflict B` | `... (its dry run conflict B, started by a helper)` |

Each dry run's log is `logs/dry_<task>_<rule>.<time>.log`.
