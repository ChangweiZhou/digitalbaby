# MiniFly research-loop kit: overview for Mr. Zhou

24 October 2026. Branch `claude/vibrant-cerf-sje726`. Written by a Claude Code session at Mr. Zhou's request, relayed by
a collaborator. No MiniFly experiment, training or historical code was run for any of this work.

## 概要（中文）

- **这是什么。** 这是给 MiniFly 研究流程本身装的"测试与自我改进系统"，不是新的 MiniFly 模型。
  - 5 个 skill 规定 Codex 和 Claude 在 探索 → 设计 → cycle 0 → 三轮测试 → 正式运行 → 审计与搜索 各阶段必须做什么、禁止做什么。
  - 检查器 `tools/minifly_check` 把散文规则变成"不通过即停"的程序检查。
  - 知识库 `PROJECT_DOCUMENTS/KB` 只追加地记录结论、事故和预测。
- **为什么。** 历史上最贵的失败都是"跑完才发现"：Package B 的控制臂失效、STOP-ONLY 的对照比较器恒为零、V6-R1 的线索与噪声之比、LEARNED-BANK 的不可学任务。共同原因是规则只是散文。
- **自我改进如何防作弊。** 检查器像 MiniFly 一样，要在"没见过的题"上考试：由看不到代码的 agent 编造新的作弊方式，先封存再测。第一批结果：可见语料 117/117，留出攻击 6/9。三处漏检随后修复，修复本身又被独立验证者打回 4 次，其中一次查出修复里写进了只在旧样本中出现的词，属于过拟合。
- **局限。**
  - 尚未在真实实验上运行。
  - 阈值是草稿，需要周先生批准（UD-0001）。
  - 工具包目前偏向加法任务，会误拦合法的分量实验（见 dogfood 报告）。
  - 散文中的拔高措辞只能靠词法提示加审阅者人工判断，无法机械保证。
- **需要周先生决定的事项**见第 7 节。

## 1. What problem this solves

The loop as practised is: explore, then experiment (design, then refinement, audit, three test runs, official run),
then audit and search. Two read-only workflows went through the 53 post-Full151 deliveries and the 1,124-document
pre-Full151 archive. They found 30 recurring failure classes; the analyses are in `KB/ANALYSIS_20261024/`. The
expensive incidents were silent and were caught late:
- Package B: a dead control arm, found after 832 receipts.
- STOP-ONLY: a zero-scoring comparator, found after a 225,792-record confirmation.
- V6-R1: a cue/noise ratio that its own formulas already implied.
- v4: a reversed teaching sign.
- v5.1: a gate that could not fail.
- LEARNED-BANK: an untestable primary task.

The rules that should have prevented these already existed as prose, for example the E0-E4 contract, and failed
anyway. The kit turns them into machine checks that block progress, placed where compute or credibility is spent.

## 2. The loop with the kit (`.claude/skills/minifly-research-loop`)

| Your step | Kit stage | Blocking command (`MFC` = `python3 -I tools/minifly_check/minifly_check.py`) | Added because of |
|---|---|---|---|
| explore | S0: read `MFC kb status`, cheapest falsifier first, neutral options with cost | none | route drift (A01-A10) |
| design | S1: ENDPOINT_LOCK (contract block as fields, predictions, lineage), AUTHORIZATION in the owner's words | `gate --stage lock` | post-hoc amendments, agent-written scope |
| refinement / audit / test run 1 | C0 testbed screen before learner code: a ceiling learner must pass and the best frozen floor must fail | `launch --stage testbed`, `score --role admission` | LEARNED-BANK T-MOD |
| test run 1 | C1 technical: one record per arm; control liveness from raw receipts | `gate --stage roster` | Package B, v4, v5.1 |
| audit | T2: executing cross-family review, cheapest-shortcut attempt, canary | REVIEW.json (REV-*, WIT-01) | self-review, padded audits |
| test runs 2-3 | C2 dev (counted as "best of k"), C3 rehearsal on a fresh world | `launch --stage dev/rehearsal`, `gate --stage official --dry` | in-cycle design search |
| official run | S6: one attempt; resume only with a determinism replay | `launch --stage official [--resume]` | reruns; LEARNED-BANK restart |
| audit and search | S7: program-owned score and verdict, generated claim and headline, findings proposed for the other family to activate | `score`, `claim`, `gate --stage publish` | claim inflation, stale pointers |

Honest exits are first-class and unpenalized: CANNOT_TEST_CLAIM_AS_SPECIFIED, INVALID (TREATMENT_NOT_DELIVERED),
NOT_ACQUIRING, FAIL.

## 3. Generalization from unit knowledge, at two levels

- **The MiniFly learner** (`.claude/skills/minifly-e3-design`).
  - Units are fixed before any stream exists.
  - The checker computes strata S0-S4 and a determinacy witness: a held item counts only if the taught units determine
    its answer.
  - Every result is compared against the strongest shortcut that uses no memory, including a FIXED_k truncation of the
    candidate's own output.
  - G = held exact own-EOS minus max(no-prior-teaching control, best shortcut, chance), reported per sample and per
    compute.
  - Only G on sealed worlds counts as program progress.
- **The research loop** (`minifly-claims-knowledge`, `minifly-governance`).
  - Findings and failure classes are the loop's "taught units".
  - Designs register predictions that cite them. Post-audit scores the predictions, and only findings that predict
    correctly on later work become "transferable".
  - Failure classes are compiled into checks. Each check is tested on held-out canaries written by agents that never
    saw the checker. That is the loop's own E3 test.

## 4. Recursive self-improvement, and why it resists gaming (`minifly-governance`)

1. **Incident to check.** Every defect becomes an INCIDENTS row. It closes only when a check and a fixture that fires
   it both exist (KB-05).
2. **Protected kit.** Changes must pass the whole regression corpus, are classified tighten, neutral or loosen, and are
   committed separately from any delivery. A loosening or a constraint change needs the owner's verbatim words with the
   numbers restated.
3. **Held-out evaluation of the kit itself.** Code-blind authors write sealed canary batches. Each batch's commitment is
   committed before the checker version it tests. A rise in visible recall without a rise in held-out recall counts as
   overfitting.
4. **Independent verification of every fix.** Verifiers write their own probes and look for loosened tests and fixes
   fitted to canary literals.

What happened today:

| Round | What | Result |
|---|---|---|
| build | 10 work packages, 3 integration rounds, 1 hardening round, each accepted only by an independent verifier | 563 tests; 117 fixtures; every check exercised (`selftest --mutate`) |
| retro | read-only scan of the whole history | 16/16 known incidents flagged; 5.9% adjudicated false flags (29.4% raw, reported) |
| held-out 1 | 12 canaries from 3 code-blind authors, against checker 0471fda3 | **6/9 attacks caught, 0/2 false alarms**, against 117/117 visible: the overfitting signal |
| self-improvement round 1 | generalizing fixes for the 3 misses (exposure by content, claim-level overclaim, commit identity) giving checker 1.0.1 (`b69cb3ae`) | independent verifiers rejected it 4 times, once for a fix fitted to a canary word. The loop was then stopped on request; 1.0.1 is green (608 tests, 156 fixtures, every check exercised) but **not verifier-accepted**. Open items are in `tools/minifly_check/INTEGRATION_LOG.md` |
| held-out 2 | 4 canaries on the claims layer against checker 1.0.1; two authors were stopped by a safety classifier and not re-dispatched, and 2 canaries were exposed to a repair agent | 2/3 attacks caught (1/2 unexposed), 1 false alarm on honest negated text, and 1 new miss (a hand-written subset verdict in STATUS.json). Too small to show improvement (`KB/RETRO/HELDOUT_20261024_R1_CLAIMS.md`) |
| dogfood | the kit applied at T0 to the completed LEARNED-BANK V1 | 7 real problems caught, but 7 false positives and 11 schema gaps: the kit is addition-shaped (`KB/RETRO/DOGFOOD_LEARNED_BANK_20261024.md`) |

## 5. What is enforced, and what is not

- **Enforced:** about 137 checks; the list and status of each are in `tools/minifly_check/RULES.json`. Planned and
  not yet enforced: UDI-01, the leave-units-out dependency index required for E3 promotion.
- **Judgment, not mechanism:**
  - Overclaiming in free prose. A lexical lint (CLM-17P) is only an aid; the T2 reviewer must adjudicate flagged
    sentences.
  - Whether a design step is the most informative one.
  - Task-shaped priors.
- **Open, stated in SPEC section 15 and section 19:**
  - In-repo identity is tamper-evident, not tamper-proof: Codex commits under the owner's git identity.
  - Owner decisions are agent-transcribed.
  - Runs outside `launch` at T1 are invisible.
  - The scorer, ladder and strata are addition-centric.

## 6. Recommended next steps

1. **Checker 1.0.2 (adoption readiness).**
   - The testbed ceiling blocker: ORACLE_FEATURES must not count as a learnability ceiling.
   - An evaluator-length component metric, and FIXED_k excluded on constant-length answers.
   - Per-world split manifests, an optional OLD pool, and a `diagnostic` launch stage so screens and surrogates count.
   - Hyperparameters passed as CLI flags included in the design identity.
   - Caps requested with a measured cost.
   - Fixes for the false-positive scanners, and the per-arm comparator LCB option.
   - SPEC 3.5 enforced mechanically: no hand-written verdict fields outside the generated CLAIM.json or VERDICT.json
     (INC-HB2-E3).
   - The LEARNED-BANK session's 16 review items (`reviews_20261024/MRK1_KIT_0471fda3.md` on that branch).
2. **Cross-family governance review of the kit by Codex** (Phase 0, step 4). Everything here was written and reviewed
   by Claude-family agents.
3. **First live T1 pilot** after 1.0.2, on technical and dev worlds only, under a grant in the owner's words with caps.
4. **First official held-out batch** written by the owner or a fresh session, sealed before the next checker merge.

## 7. Decisions only Mr. Zhou can make

1. **UD-0001: ratify or change the draft numbers** in `KB/CONSTRAINTS.json`. Write them in your own words:
   - margin_abs 0.05;
   - min_worlds_nonpilot 8;
   - old_max_drop 0.10;
   - shuf_band;
   - tier thresholds: T1 at most 4 CPU-h and 20,000 records;
   - default grant caps: 2 CPU-h and 5,000 records. The dogfood shows these block a single native history of about
     11,000 records.
2. **Whether the kit applies to the next delivery** (the schema-v1 freeze), and whether the E3 cadence counter counts
   legacy deliveries.
3. **An owner-held anchor against forged identity or decisions:** a GitHub push ruleset on the protected paths,
   owner-signed tags, or none.
4. **Agent conventions:** `Acting-Agent:` commit trailers, and each agent pushing only to its own `codex/*` or
   `claude/*` namespace.
5. **Whether sealed per-world digit-byte relabeling may be used** for E3 confirmations. It would make "no supplied
   arithmetic structure" testable, but it changes the raw-byte surface.
6. **Numeric objective targets** for E3: the held exact own-EOS level, and per-sample and per-compute budgets.
7. **Negated level mentions in CLAIM.notes** (INC-HB2-E1): allow them, or keep the ban and state it where CLAIM is
   written.
8. **Your local AGENTS.md:** merge the lines in `KB/AGENTS_POINTER.md`. A tracked root AGENTS.md was deliberately not
   created, because it would collide with your local file.

## 8. File map

- Skills: `.claude/skills/minifly-{research-loop,e3-design,claims-knowledge,independent-review,governance}/`.
- Checker: `tools/minifly_check/`. The spec is `SPEC.md`, with the integrator amendments in Section 20; field schemas
  are in `SCHEMAS.md`, the build history in `INTEGRATION_LOG.md`, and the corpus in `corpus/`.
- Knowledge base: `PROJECT_DOCUMENTS/KB/` (README, CONSTRAINTS draft, ledgers, FAILURE_CLASSES, RETRO/,
  ANALYSIS_20261024/).
- Quick start: `MFC --help`, `MFC kb status`, `MFC selftest` (about 1 minute). For a new delivery:
  `MFC init <dir> --claim-level E1|E3 --task-family ...`, then follow the stage table above.
