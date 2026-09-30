# Package A evidence intake

Status: intake complete; Muse's V2 outputs **unavailable**. A new, separately named implementation was then built
and technically qualified (see `TECHNICAL_REPORT.md`). Nothing here resumes or audits Muse's V2 run.

## Received
| File | SHA256 | Checks run here |
|---|---|---|
| `input/MINIFLY_MUSE_A_V3_CAUSAL_GATE_20260929.zip` (corrected scaffold, `science_ready: false`) | `c667ab51886e226a9e7f014af634f9e9fdb1e00ad384d0eabcfd64710972ab43` | `verify_bundle.py --integrity-only` and full `verify_bundle.py` pass (107 files; gate: 20 branch decisions correct, 22 tamper cases rejected, V2 defect rejected; raw host B `872094396ed4…`, canonical B installed). `causal_branch_gate.py --receipt` rejects Package B V2 technical receipt `minifly_b/results/technical/S0/190000.json.gz`: `N_old_rel/old_relation=288, expected 0`. |

| `input/CLAUDE_PACKAGE_A_MUSE_INTAKE_20260929.zip` | `22ef1605757d4245d4a4edc28b1bf2fbb2921ac9768cdeac1152d566897f16ca` | Manifest 30/30 files match. Contains the original V2 *input handoff* zip (`d76e54e9…`), the same V3 zip (byte-identical), protocol copies, Muse status as conversation testimony only, and a partial Codex prototype. **No Muse runner, source hashes, SPEC_LOCK/SOURCE_LOCK, receipts, stop record, P3 failure receipt or roster amendment** (`README_FOR_CLAUDE.md`). The read-only V2 audit therefore cannot be performed; V2 E1 is not asserted valid and Muse's P3 failure is not independently confirmed. |

The scaffold's `common_platform.branch_allows` differs from V2 only by the explicit five-branch map.
Its seven-candidate `CONTRACT.json` is historical input, not authorization to run P3.

## Still required before any V2 audit or roster lock (not in this session)
1. Immutable export of Muse's actual V2 source (runner, arm implementations, auditor), with hashes.
2. Muse's `SPEC_LOCK.md` and `SOURCE_LOCK.json`.
3. All Muse V2 receipts (technical and any science), logs and current status / stop record.
4. The P3 technical-failure receipt and the frozen roster-amendment record.

## Provisional Package A roster (not locked)
Candidates R1, R3, Z2, P1, P2, P4 with their declared controls (R0, R0_signed, Z0_resource, P0) and diagnostics
(R1_rand, R3_randtarget, Z2_rand), pending the amendment/technical records above. P3 stays visible as
technically unqualified, with no science comparison and no replacement; `P3_shuffle` unrun. The shared family
stays **m=26**, with P3's E1 and E3 comparisons marked unavailable.

## Package B
Not rerun. V2's 832 receipts stand: E1 valid (no candidate advantage), E3 invalid. Any E3 program would be a new
prospective multi-package experiment with its own lock and receipt namespace.

## Next steps once Muse's files arrive
Preserve them unchanged; read-only V2 audit (fact-write branches per record, birth rule, source lock, receipts);
then port the corrected branch map into a new-version runner with a per-(record, branch, store) write ledger and
an independent auditor, and run technical qualification on world 190000 only. No science.
If Muse cannot supply them: a *new* Package A built from the scaffold, labeled as such, technical qualification only.
