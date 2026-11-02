# Seal protocol (every SEALED_CONFIRM)

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`. Normative text: `tools/minifly_check/SPEC.md` section 10.
Seal file fields: `tools/minifly_check/SCHEMAS.md` section 13. The official gate runs SEAL-01..05 and CONF-01 whenever
worlds_role SEALED_CONFIRM is intended, and `MFC launch --stage official` refuses to start while any of them fails.

Principle: entropy arrives only after the lock is fixed and witnessed. Neither party can grind it, and no secret has to
be stored for long. Three parties take part: the executor, the reviewer (the other agent family) and the owner.

**Single-agent mode cannot seal.** A same-family session or subagent cannot hold the nonce, and a review debt blocks
SEALED_CONFIRM (DEBT-02). Without the other family, T2 work stays DEV or PILOT (spec 5). State that; never claim
otherwise.

## The six steps

| Step | Who | Action | Artifact and commit | Checked by |
|---|---|---|---|---|
| 1 | executor | Freeze ENDPOINT_LOCK (sha L1) and the final SOURCE_LOCK (sha L2); relay `<sha> <path>` for both | both locks committed | WIT-01 |
| 2 | reviewer | Attest L1 and L2 (reviewed_endpoint_lock_sha256, source_lock_acks), then run `MFC seal commit-nonce <dir>` | `review/SEAL_COMMIT.json` {lock shas, nonce_sha256, reviewer, utc}, committed on the reviewer's branch; the nonce stays in a private file outside the repo | SEAL-02 (reviewer namespace), SEAL-03 |
| 3 | owner | Gives GO after seeing L1 and L2 | AUTHORIZATION.stages.official {granted, verbatim, utc, lock_shas_shown}, the owner's words verbatim, committed after SEAL_COMMIT | SEAL-02, SEAL-03 |
| 4 | reviewer | Runs `MFC seal reveal <dir>` | `review/SEAL_REVEAL.json` {nonce}, committed after GO | SEAL-01, SEAL-02, SEAL-03 |
| 5 | anyone, by the tool | Derives the worlds: `MFC seal derive <dir>`, then `MFC seal verify <dir>` | the official run uses exactly the derived world ids | SEAL-04 |
| 6 | the tool, at reveal | `seal reveal` appends the CONFIRM_LEDGER row {family_id, attempt_index j, alpha_allocated} before any outcome exists | `PROJECT_DOCUMENTS/KB/CONFIRM_LEDGER.jsonl` | SEAL-05, CONF-01 |

Derivation (step 5; computed by the tool, never by hand):

```
key          = sha256(nonce || owner_go_verbatim_utf8)
world_seed_i = int(HMAC-SHA256(key, L1 || '|' || L2 || '|confirm_worlds|' || i)[:8])
```

The UDI dependent-set seed and the replay seeds use the same construction with purposes 'udi' and 'replay'.

Ordering (SEAL-02): SEAL_COMMIT precedes GO, GO precedes REVEAL, REVEAL precedes the first official output, and
SEAL_COMMIT is on a reviewer-namespace branch. SEAL-01 requires SEAL_COMMIT, the GO and SEAL_REVEAL. SEAL-03 requires that the nonce hash
matches and that the GO verbatim names the lock shas. SEAL-05 requires that the CONFIRM_LEDGER row precede outcomes.

## Duties and cautions per step

1. **Executor.** Read CONFIRM_LEDGER for the computed family before freezing: j is the number of earlier rows of the
   family, and CONF-01 requires the lock's alpha to be at most alpha_allocated. Pending the owner's decision this is
   0.05 x 2^-j (j = 0: 0.05; j = 1: 0.025; j = 2: 0.0125). OC-01 must also hold (P(pass | target) at least 0.8,
   `MFC oc <dir>`). Otherwise the run cannot be SEALED_CONFIRM.
2. **Reviewer.** Attest only shas you have read at their commits, after the delta review of the closure. Never
   commit, paste or relay the nonce, and never keep it inside the repo.
3. **Owner GO.** The executor shows Mr. Zhou both full shas and transcribes the owner's words verbatim, unchanged, with UTC
   and session. The owner's words themselves must contain both lock shas (SEAL-03): ask the owner to include them. A bare 'go' or
   'yes' does not name them. Commit the GO after SEAL_COMMIT. The reviewer must not reveal before seeing this commit.
4. **Reveal.** Merge the executor's GO commit into your branch first, so that your reveal commit descends from it,
   then run reveal and commit SEAL_REVEAL.json at once.
5. **Derive.** Use the derived world ids exactly. Never generate, screen or touch an official world before the reveal.
   AMD-04 and WLD-01 fail if an official world id appears in any earlier artifact or another delivery.
6. **Ledger.** Do not append the CONFIRM_LEDGER row by hand. It is written by `seal reveal`, before any outcome.

## Failure handling

| Failure | When noticed | What to do |
|---|---|---|
| Nonce lost before reveal | step 4 | Repeat steps 2-4: a fresh `seal commit-nonce`, a new GO after the new SEAL_COMMIT, then reveal. No run data is lost, because no official output exists yet |
| A lock changes after step 1 (a closure fix, a param change) | any later step | The attested shas no longer match: start again at step 1 with new L1/L2. An ENDPOINT_LOCK change after learner outcomes is an amendment (AMD rules); a new SOURCE_LOCK needs a fresh reviewer ack (WIT-01) |
| GO does not name both shas, or was given before SEAL_COMMIT | before reveal | Ask for a new GO after SEAL_COMMIT that names the shas; then reveal |
| GO found defective after reveal | after step 4 | The nonce is public, so a new GO could be ground against it. Redo steps 2-4 with a fresh nonce |
| Reveal committed before GO | after step 4 | SEAL-02 fails. Redo steps 2-4 with a fresh nonce |
| A redo after a reveal | — | The earlier CONFIRM_LEDGER row stays (append-only), so the next reveal gets attempt index j + 1 and half the alpha. If the lock's alpha then exceeds the allocation, CONF-01 fails; lowering alpha changes primary.* after learner outcomes, which needs a UD (AMD-02) and restarts at step 1 |
| SEAL_COMMIT not in the reviewer's namespace | any | SEAL-02 fails: the reviewer commits and pushes it on their own branch, and the executor merges it (never cherry-picks) |
| Official world ids differ from the derived ones | step 5 or later | SEAL-04 fails and launch refuses. Run official with the derived ids; nothing official may exist yet |
| Any SEAL-* failure after official outputs exist | — | Cannot happen through `launch`, which refuses. If outputs were produced outside launch, the run is not SEALED_CONFIRM (CLM-09) and is never rerun |

After the official run, the reviewer replays seal-seeded histories (SKILL.md section 8; REV-08) before any confirm
wording is published (CLM-04 requires SEAL-* and WLD-01 PASS).
