# MiniFly remaining-13 remote handoff

This handoff divides the **13 untested entries** of the bounded 16-variant inventory into two non-overlapping implementation and testing packages. It does not modify the source-locked R2/Z1/T2 run and contains **no** science receipts from it. FE1 was removed from this inventory before the three-mechanism round.

| Package | Intended operator | Variants | Count | Shared implementation work |
|---|---|---|---:|---|
| A — state and fixed-edge learning | Muse Code in a remote terminal | R1, R3, Z2, P1, P2, P3, P4 | 7 | Full151 stores, local write controls and PN→KC weight updates |
| B — topology | Claude Code in a remote terminal | T1, T3, S1, S2, S3, S4 | 6 | PN→KC graph construction, true lifetime rewiring and graph controls |

`7 + 6 = 13`; neither package includes R2, Z1, T2 or FE1. The packages share the same deterministic byte-task fixture, 64 reserved science worlds and analysis contract. Each contains a compact copy of the required source/import closure so it can be extracted on a remote machine without the Mac project tree. They contain no learned checkpoint, prior trajectories or results. Full151 is constructed from the included source; V2 anchors its newborn B matrix to exact source-of-record bytes after a cross-platform discrepancy was found.

## Readiness boundary

The 16-variant inventory and the older 17-item v2 programme are **design material, not frozen executable specifications**. In particular, many candidate update equations, thresholds, clocks, state budgets and matched controls remain unspecified. The external coding agents must complete and independently audit those locks **before** executing any science world. A technical pass is not a scientific success. If a candidate cannot be specified without an outcome-dependent choice, report `NOT_INSTANTIATED` with the blocker; do not improvise after seeing science results.

The packages are ready to hand to remote coding agents for implementation, qualification and execution. They are **not** prevalidated executables for all 13 variants. The included historical `run_science.py` targets only the existing R2/Z1/T2 experiment and must never be invoked from these packages.

## Files

- `MINIFLY_MUSE_A_7_V2_20260928.zip` and `MINIFLY_CLAUDE_B_6_V2_20260928.zip` are the current self-contained upload archives. They supersede the originals after a cross-platform B-birth mismatch was found before science.
- Each archive includes `HANDOFF.md`, `SHARED_PROTOCOL.md`, `TASK_PROMPT.md`, `CONTRACT.json`, `requirements.txt`, `verify_bundle.py`, `BIRTH_PORTABILITY_ADDENDUM.md`, an exact canonical newborn B anchor, a checksum manifest, relevant project documents and `REFERENCE_SOURCE/`.
- `build_packages.py` regenerates both archives from the source-lock's **source files only**, excluding current run outputs and old trajectories.

## Remote execution

Use a persistent remote filesystem and a terminal with Python 3.11 and all pinned dependencies in `requirements.txt`. Unzip each package into a **different** workspace. Read `BIRTH_PORTABILITY_ADDENDUM.md` first. Run `python verify_bundle.py` before any implementation; it checks every bundled file and the complete newborn state, anchors B exactly and constructs a technical fixture but runs no trajectory. `python verify_bundle.py --integrity-only` checks hashes and Python syntax before dependencies are installed. The agents must not execute science until they have written exact candidate specs, tests, auditors, a resource estimate and a new package-local source lock.

Muse's browser assistant should only be used if it offers such a persistent terminal. Meta documents [Muse Code](https://dev.meta.ai/docs/muse-code) as its terminal/CI coding agent; the zip requires no Muse-specific API. The birth anchor makes the starting B matrix portable but does not guarantee bit-for-bit parity of later numerical trajectories. The two operators should return separate result bundles for independent review. Their results may be compared descriptively across packages because the fixture/worlds match, but the current R2/Z1/T2 run used a different roster and cannot be folded into a 16-way paired interval without a separately designed analysis.

Nothing in this handoff asks the Mac to run the remaining 13 science trajectories.
