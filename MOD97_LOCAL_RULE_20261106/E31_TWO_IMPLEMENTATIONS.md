# E31: two independent implementations under one file name

Workers A and B each wrote `arch31.py` and `exp31_switch.py` without reading the other's, as the brief required. On the default branch:

- `arch31.py`, `exp31_switch.py` are **worker A's** (branch `claude/sleepy-knuth-q96dd1` at a930ed95). They made the lives of arms G, H and P2b.
- `arch31_workerB.py`, `exp31_switch_workerB.py` are **worker B's** files of the same names (branch `claude/charming-clarke-u9g850` at 2bd86f84), renamed here only so both can sit in one folder. They made the lives of arms K and P2a. To rerun one of those lives, check out worker B's branch, where the files carry their original names and the hashes in the receipts.

Nothing inside either file was changed.
