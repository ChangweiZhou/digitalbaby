# MiniFly reasoning scaling engineering cycles

The [revised design](REVISED_DESIGN.md) records the benchmark changes. The [cycle report](CYCLES.md) records what was exercised, outcomes, and the remaining requirements for scientific launch.

The harness imports the archived V216 learner and sensor through a supplied archive path. It verifies their hashes and the fixed sensory assets against the V216 campaign lock. Use the Python environment that runs the archived V216 dependencies. The rehearsal uses one technical family and four arms.

Run the commands below from this directory. The archive directory contains the V216 and V181 packages and Package A sensory assets. Each command must finish before the next begins.

```sh
archive=/path/to/archived/digitalbaby
python_bin=/path/to/v216/python
scratch=$(mktemp -d)
"$python_bin" -B cycles.py "$archive" 1
"$python_bin" -B cycles.py "$archive" 2
"$python_bin" -B cycles.py "$archive" 3 "$scratch/checkpoint.pt"
"$python_bin" -B cycles.py "$archive" resume "$scratch/checkpoint.pt"
```

Cycle 3 writes a restart-pending receipt. The last command checks the saved checkpoint hash and exact next-update state in a fresh process, then finalizes that receipt. The harness has no main-roster launch mode.
