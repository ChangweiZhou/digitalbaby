# Claude handoff: Package A causal-branch repair scaffold (V3)

**Status:** This is a corrected *input scaffold*, not Muse's current science
runner and not a source-locked, science-ready replacement. It was derived from
the untouched `MINIFLY_MUSE_A_7_V2_20260928.zip`. No old or new science world
was run to make it. The only executable reference-source change is the explicit
five-branch mapping in `REFERENCE_SOURCE/.../common_platform.py`; the verifier
and independent technical gate are new. The original V2 package and all
committed receipts must remain immutable.

## What went wrong

V2's `branch_allows` constructed `N_old_relation` and `N_new_relation`, but the
actual branch IDs are `N_old_rel` and `N_new_rel`. It therefore allowed every
relation write in both alleged no-write branches. In Claude Package B's 832
committed receipts, `N_old_rel.old_relation=288` and
`N_new_rel.new_relation=144`, equal to W. The same source was shipped to Muse.
This makes V2 E3 unidentifiable; its zero contrast is a failed instrument, not
a scientific null. Correctly withheld fact writes can support an E1-only
analysis after separate birth, lock and receipt audits.

## Literal gate for the frozen 600-record fixture

All values are *store writes*, not teaching-record counts. Each branch must
have exactly these four aggregate counts in its full-life technical receipt:

| branch | old_fact | old_relation | new_fact | new_relation |
|---|---:|---:|---:|---:|
| W | 768 | 288 | 768 | 144 |
| N_old_fact | 0 | 288 | 768 | 144 |
| N_old_rel | 768 | 0 | 768 | 144 |
| N_new_fact | 768 | 288 | 0 | 144 |
| N_new_rel | 768 | 288 | 768 | 0 |

`causal_branch_gate.py` derives neither this matrix nor its receipt check from
the runner. It also checks that all 600 presented records and teacher times
match across branches, rejects a copy of the V2 defective function, and
rejects 22 deliberate ledger/exposure corruptions. It is wired into
`verify_bundle.py`, so `--integrity-only` fails if the reference source
regresses. This *does not* replace running a full technical life: the script
must be called with each package-local arm's technical receipt.
The inherited receipt contains aggregate writes but no per-record write bitmap.
The new Muse/Claude implementation must record the write decision for each
`(record, branch, store)` and independently verify it against the frozen
fixture; aggregate totals alone could conceal a wrong write compensated by a
missing write elsewhere.

## Instructions for the takeover

1. Obtain Muse's **actual V2 source, source hashes, receipts, SPEC_LOCK and
   current stop/status record**. This archive cannot supply those files. Preserve
   them unchanged; record any other defects (birth, lock, arm-specific controls)
   independently of the relation-branch defect. Do not claim this scaffold
   validates code that has not been inspected.
2. Compare Muse's branch decisions and all auditor imports with this corrected
   reference. Change the *new version* of the runner explicitly; its independent
   auditor must use the literal fixture matrix or independently count events,
   never import the runner's `branch_allows` as its sole oracle. Add a deliberate
   mutation that restores 288 old-relation writes and prove it fails. Also
   mutate one per-record write decision while holding its aggregate count fixed
   by an opposite mutation elsewhere; the auditor must reject that pair.
3. Run `python verify_bundle.py --integrity-only` in the extracted scaffold.
   On the pinned runtime also run full `python verify_bundle.py`. After porting
   Muse's actual arms, run world 190000 as a complete **technical** life for
   every arm and check each receipt with
   `python causal_branch_gate.py --receipt PATH/TO/TECHNICAL.json[.gz]`.
   Verify both old and new relation lesions, all birth and source fingerprints,
   clone isolation, E1 controls, arm-specific invariants and deliberate tamper
   rejection before writing a new source lock. A final independent audit must
   also scan every newly committed science receipt with
   `python causal_branch_gate.py --receipts-dir PATH/TO/NEW/SCIENCE`.
4. Resolve the roster explicitly. Muse reported P3/P3_shuffle unqualified and
   an amended 13-arm plan; the V2 seven-candidate contract in this scaffold is
   historical, **not permission to restore P3 or silently change m=26**. Keep
   missing candidates visible. A new scientific protocol must state the roster,
   primary family and which comparisons are unavailable.
5. Do not resume under V2's source lock or overwrite a committed world-arm.
   Decide separately whether any corrected science execution is justified, with
   a new version, lock, receipt namespace and explicit treatment of the old
   E1-only evidence. Do not automatically restart after an integrity failure.

## Scope of the verification supplied here

The source truth table, synthetic receipt and mutation tests pass without
NumPy or running a trajectory. The gate rejects a committed Claude Package B
receipt with the defective relation ledgers. Full Muse-arm technical
qualification and scientific acceptance **remain unverified** until its actual
source and technical receipts are available.
