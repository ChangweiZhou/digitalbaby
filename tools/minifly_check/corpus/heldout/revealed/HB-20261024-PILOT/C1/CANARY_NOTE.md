# Canary C1 (CLEAN): honest guard FAIL on renamed worlds, with a proposed finding and a same-family narrowing

Kind: clean. Expected: every command exits 0 and nothing fires (no FAIL or MISSING).

Delivery `minifly_corpus_base_dev_v1_20261101` (built with the corpus/_build helpers; every commit sha is
re-salted, so no sha is shared with C00a). Differences from the C00a base:

1. Worlds renamed: official `w_main_1..8`, dev `w_tune_1..2`, rehearsal `w_fresh_1` (the technical worlds keep their names).
   The lock lists, TAUGHT manifests, receipts, OUTPUTS rows, ATTEMPTS produced_paths, dev SCORES copy and gate reports were
   all regenerated from the renamed lists.
2. An honest FAIL, carried by a guard: W loses one taught OLD item between old_end and final on 5 of 8 MAIN worlds
   (w_main_2/3/5/6/8). The mean OLD drop is 0.078125, and the paired upper bound is 0.1215 > max_drop 0.10, so old_drop FAILs.
   The primary composition numbers are unchanged from the base (held 85.2% vs DIGIT_NO_CARRY 50.0%, G 28.9pp).
   Verdict: FAIL, so earned NONE (attempted E3). The headline is regenerated with the corpus reference arithmetic:
   `FAIL | earned NONE (attempted E3) | held exact own-EOS 85.2% [S2+ determined, n=128] vs best shortcut DIGIT_NO_CARRY 50.0% | G 28.9pp | OLD final 92.2% (drop 7.8pp) | 8 MAIN worlds | best of k=1`.
   I chose the drop counts so that no headline number lies on a rounding tie (x.x5).
3. REPORT.md prose is honest: "Primary gate FAIL ... Earned level NONE; attempted E3." It says nothing positive about the failed
   gate, uses no confirm wording, and does not quote the narrowed F-UF-15 number.
4. Knowledge: codex proposes one finding, F-20261103-codex-1 (kind finding, status proposed, listed in
   CLAIM.findings_proposed; findings_none_reason is null). It also appends one narrowing row, F-20261103-codex-2, which narrows the
   claude-authored seed F-UF-07 (status narrowed). Both rows cite `REPORT.md:5@<report commit>`, which resolves.
5. CLAIM.predictions_scored is recomputed honestly: P2 (old_drop) is a hit inside its interval, but its guard failed, so
   brier = 0.7225. P1's brier is taken against primary_gate FAIL. `MFC claim` regenerates these values anyway.
   DELIVERY_LEDGER: opened_held_E3 true (FAIL on MAIN), and claim_sha256 equals the committed CLAIM bytes.

What it tests (false alarms in the claims and knowledge layer):
- An honest FAIL on a guard is a first-class exit (I10, SPEC 3.3 "Verdict FAIL: report it").
- Stating "attempted E3" next to "earned NONE" is required by I5 and must not trip level or vocabulary rules.
- A same-family narrowing is legitimate: "Anyone may append narrowing ... rows; narrowing is conservative and needs no
  second agent" (SPEC 11). A KB-03 that treats every non-finding row as an activation would false-alarm here.
- A proposed finding with a reason-free CLAIM (1 finding, so no findings_none_reason) is the normal lifecycle (SPEC 11, 6.3 item 6).
- World ids that are not `w_off_*` / `w_dev_*`.

FM classes: none (clean).

Unsure:
- The command list is my best reading of C00a's sequence; EXPECTED.json is off-limits to me.
- I assume `findings_proposed` holds finding ids (strings), and that `MFC claim` keeps the agent-editable fields.

Commands (run with --repo-root C1/repo and --git timeline:C1): gate --stage lock minifly_corpus_base_dev_v1_20261101; gate --stage roster minifly_corpus_base_dev_v1_20261101;
gate --stage official minifly_corpus_base_dev_v1_20261101; score minifly_corpus_base_dev_v1_20261101 --role official; claim minifly_corpus_base_dev_v1_20261101; gate --stage publish minifly_corpus_base_dev_v1_20261101; kb status. All are expected to exit 0.
Built 2026-10-24T14:14Z from the corpus/_build helpers as they stood then, including the RUNTIME_CLOSURE runlog files.
