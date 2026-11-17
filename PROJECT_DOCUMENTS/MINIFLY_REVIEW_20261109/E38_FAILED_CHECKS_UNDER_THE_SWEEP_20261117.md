# E38: the two checks that failed under the sweep (17 November 2026)

This note examines one open point of the record: item 27 of `COORDINATION_20261117/PLAN.md` (line 48), "E38's two checks that failed under the sweep and have no ruling on file". It is a reading of files and a set of computations in numpy on saved weights. No network was trained, torch was not used, no code of another branch was run, and the only files written are this note and `e38_checks_read.py` beside it.

## 0. The question and the answer

**The question.** Experiment E38 specified two checks of its one new piece of code, a step that pulls each weight toward an earlier value. Under one of the two learning rules both checks failed. Was the code wrong, was the check wrong, and do the results of E38 under that rule stand?

**The answer.**

1. The step is not shown wrong by anything on file, and several things on file exclude the errors the checks were written to catch. The checks failed because they were specified in a way that cannot be expected to pass under that rule. They ask two computations that are equal up to rounding to stay equal to within 1e-5 and 1e-6 after 64 updates. Under exact back-propagation that holds. Under the sweep the update depends on the sign of every weight, so it changes by a finite amount as soon as one weight that lies next to zero has one sign in one computation and the other sign in the other. This is possibility (b) of the task.
2. The parts of the E38 report that would be unsafe if the step were wrong under the sweep are the two arms with a pull under the sweep and what is read from them: their curves, their marks, reading 5 under the sweep, and predictions P38.3 to P38.6 and P38.8. Everything under back-propagation and the three sweep arms without a pull are safe either way.
3. All 20 committed weight files of E38 reproduce the receipts exactly: 200 counts, largest difference 0. The weights of the two pulled arms lie nearer the birth than the unpulled arm by the same factors under the sweep as under back-propagation (0.59 and 0.44 at strength 0.25; 0.24 and 0.17 at strength 2.5), on every pulled tensor.
4. Proposed ruling: the results under the sweep stand, with the qualification stated in section 6. No life has to be run again.

## 1. Terms and names

| Term | Meaning here |
|---|---|
| D | `origin/codex/minifly-consolidated-20261023` at 7cabb13a, the default branch. Every file of D was read with `git show 7cabb13a:<path>` |
| V | the working tree of `claude/vibrant-cerf-sje726`, folder `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/`, on 17 November 2026 (HEAD 92f82b3b when this note was written) |
| M | the folder `MOD97_LOCAL_RULE_20261106` of D |
| the network | a recurrent network of gated units (GRU) that reads an item byte by byte: a byte table (`book`, 258 rows of 8 numbers) and six linear maps, `l1` (8 to 512), `rnn.weight_ih_l0` (512 to 768), `rnn.weight_hh_l0` (256 to 768), `lm` (256 to 512), `l2` (512 to 512), `l3` (512 to 257). 1,124,113 numbers, of which 1,122,049 are outside the byte table |
| update | one step of the optimizer on one batch of 64 items |
| rule B | exact back-propagation: the error is passed back through each linear map by the map's own weights |
| rule S, the sweep | `local=True` in the code: the forward computation is the same; the error is passed back through each linear map by `sign(W) * mean|W|` in place of `W` (`exp3_state.SignLinear`). Given the error at its output, each map's own weights are changed as under rule B |
| the optimizer, its decay | AdamW at rate 0.001. With decay d it multiplies every weight outside the byte table by `1 - 0.001 * d` before each step. The standing value of d is 0.25 |
| birth, life | a birth is a network after 24,576 updates on the first family; a life continues from the birth file on later tasks. E38's lives are 12,288 updates on a second family alone |
| family | four operations (add, sub, max, min) on two operands under one modulus, with code bytes of its own. The first family has modulus 89, the second (F1) modulus 97 |
| reduction pairs | the brief's "wrap pairs": pairs (a, b) with a + b at least the modulus for add, or a − b below zero for sub, so that the answer needs the modulus. 3,916 of the 7,921 pairs under 89, for each of add and sub |
| the pull | the brief's "anchor": before each optimizer step, `p -= 0.001 * k * (p - p_birth)` on every weight outside the byte table, with the optimizer's own decay set to 0. k is the strength: 0.25 in arm `anchor`, 2.5 in arm `anchor10` |
| arms | `zero` (decay 0.25 toward zero, as E26), `none` (no decay), `anchor`, `anchor10`, `rehearse` (decay as `zero`; items of the first family make up about half of the stream) |
| row | the number of updates at which a life was scored and its weights saved |
| receipt | a JSON file written by the runner with settings, hashes and scores |
| relative difference | of two tensors of the same name: the largest absolute difference over the largest absolute entry (`exp38_erase.max_rel`, lines 311 to 313). It is the quantity both checks compare with their tolerance |
| float32, float64 | numbers of 32 and of 64 bits. One rounding changes a float32 by a relative 6e-8 at most and a float64 by 1e-16. The runner computes in float32 |

## 2. What the record holds

### 2.1 The two checks, in the brief's words

D:`OPUS_TASK_E38_WHAT_ERASES_THE_BIRTH_FAMILY_20261115.md` (revision 2):

- Line 42: "torch's AdamW shrinks the weight before the Adam step, so the pull goes before the step too."
- Line 65: "2. **The anchor step against torch's own decay.** With `birth` set to zeros and KAPPA = 0.25, the weights after 64 updates agree with arm `zero`'s (largest absolute difference at most 1e-5 of each tensor's largest entry). The same with KAPPA = 2.5 against a run whose optimizer keeps its own decay at `weight_decay = 2.5`."
- Line 66: "3. **The anchor step in a second form.** A second implementation, written separately as `p.copy_(p0 + (1 - LR * KAPPA) * (p - p0))`, gives weights after 64 updates that agree with the first form to 1e-6 relative, for both strengths. Checks 2, 3 and 4 together stand in for a second worker's implementation of a three-line step; the audit agrees that with them none is needed."
- Line 64, on check 1 only: "If not, the runner is wrong: stop." The brief names no consequence of a failure of check 2 or 3.

The second sentence of check 2 and the whole of check 4 were required by the reviewer's audit (V:`AUDIT_E36_E37_E38_BRIEFS_20261115.md`, lines 78 to 101: "**3.2 Checks 2 and 3 catch most errors, and two slip through.**" ... "With these two, a second worker's implementation is not needed."). The audit's table lists the errors that check 2 catches: wrong sign, wrong parameter set, the optimizer's own decay left on, the factor of the rate missing, the pull applied to copies. The audit also wrote (line 173) that it had not checked "That torch 2.15's AdamW applies its decay before the Adam update. torch is absent here; check 2 of E38 tests it." Neither the brief nor the audit says that the two computations compared are not identical bit for bit, or asks what a difference of rounding size becomes after 64 updates under either rule.

### 2.2 What the worker ran

D:M/`exp45/e38/run_stage1.sh`, line 7: `for r in S B; do for c in 2 3 4 5; do python3 exp38_erase.py check $c $r 73129; done; done`. D:M/`exp45/e38/logs/checks.log`, lines 1 to 8: "check 2 S FAIL", "check 3 S FAIL", "check 4 S pass", "check 5 S pass", "check 2 B pass", "check 3 B pass", "check 4 B pass", "check 5 B pass".

D:M/`exp38_erase.py`, lines 326 to 342, is the code of the two checks. Check 2 runs four lives of 64 updates from the birth file: arm `zero`; the pull to zeros at strength 0.25; the optimizer's own decay at 2.5; the pull to zeros at 2.5. It compares the first with the second and the third with the fourth. Check 3 runs the pull to the birth values in its two forms at each strength and compares them. Both use `max_rel` on all 13 tensors and the tolerances of the brief. This is what the brief specifies.

After the failure the worker wrote and ran a diagnostic that the brief did not ask for, D:M/`e38_check2_diag.py`: the same four comparisons with the weights recorded after updates 1, 2, 4, 8, 16, 32 and 64, once in float32 and once with every tensor in float64. Its receipts are D:M/`exp45/e38/e38_check2_diag_S_73129_float32.json` and `..._float64.json`. It was run under the sweep only.

The ten receipts of checks 1 to 5, both diagnostic receipts and all ten life receipts record the same hash of the runner, `exp38_erase.py` sha256 a84b834d02d5... (22 receipts; part A of the script). The code that passed under rule B is the code that failed under rule S.

### 2.3 What failed, and by how much

From the receipts D:M/`exp45/e38/e38_check{2,3}_{S,B}_73129.json` (part A of the script):

| Check | Rule | Comparison | Tolerance | Largest relative difference | Tensors beyond the tolerance | Largest among the other tensors | Result |
|---|---|---|---|---|---|---|---|
| 2 | S | pull to zero at 0.25 against the optimizer's decay 0.25 | 1e-5 | 5.16e-4 (`l1.bias`) | 4 of 13: `l1.bias` 5.2e-4, `l1.weight` 8.7e-5, `l2.weight` 7.1e-5, `rnn.weight_ih_l0` 2.3e-5 | 1.8e-6 | fail |
| 2 | S | the same at 2.5 | 1e-5 | 4.03e-4 (`l1.bias`) | the same 4: 4.0e-4, 4.2e-5, 4.5e-5, 1.1e-5 | 5.2e-7 | fail |
| 2 | B | at 0.25; at 2.5 | 1e-5 | 2.85e-6; 4.65e-7 (`l2.weight`) | none | | pass |
| 3 | S | two forms of the pull to the birth, 0.25 | 1e-6 | 4.19e-5 (`l1.weight`) | 4 of 13: `l1.weight` 4.2e-5, `l1.bias` 2.5e-5, `l2.weight` 1.6e-5, `rnn.weight_ih_l0` 6.0e-6 | 4.6e-7 | fail |
| 3 | S | the same at 2.5 | 1e-6 | 7.76e-3 (`l2.weight`) | all 13, from 8.7e-6 (`l3.bias`) to 7.8e-3 | | fail |
| 3 | B | at 0.25; at 2.5 | 1e-6 | 6.05e-7; 5.04e-7 (`l2.weight`) | none | | pass |

Under rule B the passes are not wide: 2.85e-6 against 1e-5, and 6.05e-7 against 1e-6.

The diagnostic, rule S, largest relative difference over the 13 tensors after each recorded update:

| Pair | float32: 1 | 2 | 4 | 8 | 16 | 32 | 64 | float64: largest at any update |
|---|---|---|---|---|---|---|---|---|
| decay 0.25 against pull to zero | 9.9e-8 | 1.4e-6 | 1.4e-6 | 1.5e-6 | 1.5e-6 | 1.5e-6 | **5.2e-4** | 3.2e-15 |
| decay 2.5 against pull to zero | 8.4e-8 | 5.7e-7 | 5.8e-7 | 5.8e-7 | 5.9e-7 | **3.2e-4** | 4.0e-4 | 5.9e-15 |
| two forms, 0.25 | 0 | 5.0e-8 | 1.6e-7 | 1.6e-7 | 2.4e-7 | 3.8e-7 | **4.2e-5** | 9.5e-16 |
| two forms, 2.5 | 0 | 5.0e-8 | 1.0e-7 | **5.9e-4** | 1.6e-3 | 4.6e-3 | 7.8e-3 | 1.6e-15 |

In each float32 row the difference stays at the rounding level and then rises by two to four orders of magnitude between two recorded updates (bold). At the first recorded update after the rise the tensors beyond the tolerance are `l1.bias` and `l1.weight`, with `l2.weight` and `rnn.weight_ih_l0` in three of the four rows; the others are at most 9.0e-6 (second row) and at most 1.8e-6 (the other three rows).

### 2.4 What the worker wrote

D:M/`exp45/e38/E38_REPORT.md`:

- Lines 14 and 15, the table of checks: "**FAIL** (largest 5.2e-4, l1.bias)" and "**FAIL** (κ 0.25: 4.2e-5; κ 2.5: 7.8e-3)" under S, "pass" under B.
- Line 21: "**Checks 2 and 3 fail under the sweep as specified.** I report them as failed and do not reinterpret them as passes. A diagnostic separates arithmetic from rounding (`e38_check2_diag_S_73129_float{32,64}.json`). With every tensor in float64, the two pairs agree to 2e-16 to 6e-15 at every update to 64, for both strengths. That is rounding level, so the pull is the same arithmetic as torch's decay, and its two forms are the same arithmetic as each other. In float32 the largest difference is 1e-7 after one update and stays near 1e-6. It then jumps by two to four orders of magnitude at single updates: for example, from 1.5e-6 at update 32 to 5.2e-4 at 64 for κ 0.25, and to 5.9e-4 at update 8 for the two forms at κ 2.5. The undecayed embedding table also differs (1e-7). Under backprop the same comparisons pass. The pattern fits rounding-level differences tipped into discrete flips by the sweep's sign-based way back. It does not fit a wrong step. The brief offers checks 2 to 4 in place of a second worker's implementation. Under the sweep they stand only with this diagnostic, which the brief did not specify; a reviewer should decide whether that suffices."
- Line 23: "Check 1 passed, so the stop condition (reading 0) does not apply."
- Line 99, under "Limits": "Checks 2 and 3 failed as specified under the sweep. The float64 diagnostic indicates rounding, not a wrong step, but it is my own diagnostic, not the brief's, and no second worker has reviewed it."
- Line 101: "No independent review of this run has taken place. Nothing here is confirmed, sealed or adopted."

The figures the worker quotes are the receipts' figures (part A). One wording is loose: "at single updates" is an inference, since the diagnostic records seven updates and not every one.

### 2.5 What the coordinator and the reviewer wrote

D:`MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md`:

- Line 2959 (section 9.134, when the brief was drafted): "In place of a second worker's implementation of the three-line anchor step I specified two checks: against torch's own decay with the anchor at zero, and against a second written form."
- Line 2980 (9.135): "The audit found every testable fact in the three briefs to hold (stream seeds, the saved feedback matrices, the anchor code, the rehearsal stream, ...)".
- Line 3014 (9.136), before the report was complete: "If the complete report confirms it: decay is not the cause, and a uniform pull toward the birth weights does not protect."
- Lines 3104 and 3106 (9.141, the merge of E38): "The tables are the workers'; the reviewer states its reader reproduces all fifteen E40 files at row 8,192. I recomputed none of them this time and say so." and "**E38 (seed 73129): what erases the unrehearsed family.** Under both rules, at row 12,288: with decay, without it, and with the anchor at both strengths, the birth family's wrap pairs are at 0.0 to 0.2% while the new family is at 86 to 97; with rehearsal they are at 96.8 and 93.9 and the new family at 92.4 and 92.9. So decay is not needed for the loss (reading 2), a uniform pull toward the birth weights does not protect (reading 5), and plain rehearsal keeps and acquires."

Section 9.141 does not mention checks 2 and 3, the diagnostic, or the worker's request that a reviewer decide. No line of the review after line 2959 speaks of a check of E38: the two later lines that hold both "E38" and the word are about E39's check 6 (line 3045) and about weight decay, which "can be checked now" (line 3203).

The reviewer's documents in V read E38's figures as the worker's and do not mention the checks: `MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md` (revision 22), line 584 ("E38's rows are not committed ...; their figures below are the workers'"), lines 223, 591, 631 and 1536; `DISCUSSION_MEMO_20261116.md`, section 5.3, line 73 ("These are the worker's figures."). The reviewer's audit of the brief is of 15 November and precedes the run.

### 2.6 Whether any later document rules on it

None was found. The search: `git grep -l -i -E "check2_diag|float64 diagnostic|checks 2 and 3|check 2 and 3|checks 2, 3"` over the `.md` files of every local and remote ref (42 refs), and for the two local refs and the refs under `origin/claude/` up to `vibrant-gauss-tops5f` also over `.py`, `.json`, `.txt`, `.log` and `.out`. The matches are the E38 brief, the E38 report, the diagnostic and its two receipts (the same files on the branches that contain D's history), the E48 brief and report (their own checks 2 and 3), and in V the audit of the brief, a prompt quoted in `DISCUSSION_MEMO_20261115.md` (line 1659: "do checks 2 and 3 suffice in place of a second implementation"), `AUDIT_E39_SMALL_WORLD_PROTOCOL_20261115.md`, `E48_RESULT_VERIFICATION_20261117.md` and `e48_result_verify.py` (other experiments' checks), and the inventory. A second search of D's `.md` files for "E38" within 300 characters of "rounding", "failed", "fails", "diagnos", "second implementation" or "second worker" gives review line 2959 and report lines 5 and 21 only. The inventory states the same (V:`COORDINATION_20261117/INVENTORY_DENSE_LINE_20261117.md`, lines 472, 473, 607 and 634: "not found: no ruling in the review or in the memo").

**A ruling on a like case exists, for another experiment.** D:`OPUS_TASK_E31_ONE_SWITCH_LIVES_20261113.md`, addendum 2 of 14 November, line 92: "1. **G: accepted as agreeing.** Both evaluate the rule the brief wrote; one uses two matrix products and one a stacked matrix, and float32 rounding (about 7e-7 per crossing) compounds into a different hash. From here on, agreement between the two implementations means equal hashes, **or** gradients on one common batch within 1e-6 relative when each worker runs the other's function." The review records it at line 2732 (9.123). The E38 brief was written the next day and compares weights after 64 updates.

## 3. Why the checks failed

### 3.1 The code

The step of a life, D:M/`exp38_erase.py`, lines 82 to 91:

```
def step(m, opt, items, local, pull=None):
    x, y, mask = X.batch_tensor(items)
    logits = arch26.forward(m, x, local, True)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    if pull is not None:
        pull()
    opt.step()
    return float(loss.detach())
```

It is D:M/`exp26_candidate.py`, lines 84 to 91, with the two lines of the pull added. The pull, lines 94 to 105: form 1 is `p.sub_(LR * kappa * (p - p0))` and form 2 is `p.copy_(p0 + (1 - LR * kappa) * (p - p0))`, each under `T.no_grad()` over `zip(decayed, birth)`. `decayed` is `list(opt.param_groups[0]["params"])` (line 164), the live weights outside the byte table. `birth` is `[p.detach().clone() for p in decayed]` (line 170), taken after the birth file is loaded and before any update. For the three arms `none`, `anchor`, `anchor10` the optimizer's decay of that group is set to 0.0 (lines 167 to 168). A pull function is made only when the arm's target is the birth (lines 169 to 173); in `zero`, `none` and `rehearse` the variable `pull` is `None` and the two added lines do nothing.

Reading these lines I find no error. The pull is applied to the live weights, to the 12 tensors of group 0 and not to the byte table, with the factor of the rate, with the right sign, after `backward()` and before `opt.step()`. The gradient that `opt.step()` then uses was computed at the weights before the pull, as with the optimizer's own decay. Nothing in the pull or in `step` reads the rule: the only place where `local` enters is `arch26.forward`.

The sweep, D:M/`exp3_state.py`, lines 91 to 108:

```
class SignLinear(T.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        return linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g):
        x, weight = ctx.saved_tensors
        S = weight.sign() * weight.abs().mean()
        gx = g @ S
        gw = g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        gb = g.reshape(-1, g.shape[-1]).sum(0)
        return gx, gw, gb


def lin(x, layer, local):
    return SignLinear.apply(x, layer.weight, layer.bias) if local else linear(x, layer.weight, layer.bias)
```

D:M/`arch26.py`, lines 50 to 75, sends every one of the six linear maps through `lin(., ., local)`; the map from the recurrent state is crossed once for every position of the item. `gx`, the error passed on to the map's input, contains `weight.sign()`. That function is not continuous at zero: a weight of +1e-9 and a weight of −1e-9 give errors that differ by `2 * mean|W| * g`, however near to each other the two weights are. Under rule B the error passed on is `g @ W`, which changes by 2e-9 times `g`. This is the only place where the two rules differ, and it is where a difference of rounding size between two computations can become a difference of finite size.

### 3.2 The three possibilities and what each predicts

(a) A defect of the pull under the sweep. The errors the audit lists act on every pulled tensor from the first update, in proportion to the strength and to the number of updates. In check 2's statistic after one update they give 2.5e-4 or more at strength 0.25 and 2.5e-3 or more at 2.5 (computed in part D5 from the birth file; table in 3.4). They would also fail under rule B, since the code is the same.

(b) A check that assumes what holds only for the exact gradient. Then: the pair agrees at the rounding level for some updates and parts at one update; the size of the parting does not grow with the strength; it appears first in the tensors that lie behind the map whose weight differs in sign; in float64, where rounding is a thousand million times smaller, no parting occurs in 64 updates; under rule B the same code passes.

(c) Something else, for example an amplification of rounding that is continuous and large under the sweep. Then float64 would show the same growth factor as float32, a few thousand between update 1 and update 64.

### 3.3 What the receipts show

Every prediction of (b) is met and those of (a) and (c) are not.

- Before the rise the pair agrees at the rounding level in float32 under the sweep: 9.9e-8 and 8.4e-8 after one update, at most 1.5e-6 up to update 32 (strength 0.25) and at most 5.9e-7 up to update 16 (strength 2.5) for check 2's pair; at most 3.8e-7 up to update 32 and 1.0e-7 up to update 4 for check 3's pair. An error of the audit's list would give 2.5e-4 or more after the first update.
- The failed values do not scale with the strength: 5.2e-4 at 0.25 and 4.0e-4 at 2.5 in check 2.
- In three of the four comparisons the tensors beyond the tolerance are the same four, `l1.bias`, `l1.weight`, `l2.weight`, `rnn.weight_ih_l0`, and the other nine are within the tolerance. `l1` is the map whose error comes back through the signs of `l2.weight` and `rnn.weight_ih_l0`. Those two matrices hold 21,582 of the 27,674 weights of the six maps that lie within 0.001 of zero at the sweep birth (part D5).
- For check 2's pair the growth from update 1 to update 64 in float64 is a factor of 15 and of 30, and the largest value is 5.9e-15. In float32 the same pair grows by a factor of about 5,000. For check 3's pair, from update 2 to update 64, the factors are 6 and 16 in float64 and 840 and 156,000 in float32. A continuous amplification would give the same factor in both precisions.
- Under rule B both checks pass, with the same file.

### 3.4 Computations in numpy on the first two batches of the life

Part D of `e38_checks_read.py`. The batches are rebuilt from the stream (`exp16_integrated.entry_stream(family_entries("F1", seed), seed + 101)`); my first 256 batches have the digest that the worker's dry life records (D1). The gradient is my own backward pass through `arch26.forward`, with a switch between the two rules. Against central differences on 52 entries the exact gradient is right to 2.9e-9 of each tensor's largest entry (D2). The weights are the sweep birth, `exp45/e26/e26_dev_S_73129.pt`, except where rule B's birth is named.

**A difference of rounding size that changes no sign does nothing unusual under either rule (D3).** Every weight changed by a relative 1e-7 with all signs kept: the gradient changes by 1.0e-6 under rule B and by 3.7e-7 under rule S, relative to each tensor's largest entry.

**One weight next to zero given the other sign changes the sweep's gradient by a finite amount and the exact gradient by nothing measurable (D4).** Batch 1, float64, the weight of smallest absolute value in each matrix:

| Weight changed | Its value | Rule B: largest relative change of any tensor's gradient | Rule S: tensors whose gradient changes by more than 1e-5 |
|---|---|---|---|
| `l2.weight[28, 227]` | −4.3e-8 | 1.6e-9 | `l1.weight` 1.5e-3, `l1.bias` 1.0e-3, `book.weight` 1.6e-5; all others at most 3.2e-9 |
| `rnn.weight_ih_l0[427, 236]` | −1.7e-8 | 2.4e-9 | `l1.bias` 8.8e-3, `l1.weight` 5.6e-3, `book.weight` 6.4e-5; all others at most 6.3e-10 |
| `rnn.weight_hh_l0[110, 253]` | −1.4e-7 | 6.0e-9 | `rnn.weight_ih_l0` 1.2e-3, `rnn.bias_ih_l0` 7.2e-4, `rnn.weight_hh_l0` 5.9e-4, `rnn.bias_hh_l0` 3.7e-4, `book.weight`, `l1.weight`, `l1.bias` 1.2e-4 to 2.2e-4 |
| `lm.weight[150, 139]` | +7.9e-7 | 2.6e-7 | the recurrent tensors 2.1e-3 to 7.4e-3, `l1` 5.9e-4 and 7.4e-4, `book.weight` 4.4e-4 |

The first two rows are the pattern of the receipts: a change of sign in `l2.weight` or `rnn.weight_ih_l0` reaches `l1.weight` and `l1.bias` and nothing else but the byte table. (The smallest weights of `l3.weight` and `l1.weight` are 2.0e-5 and 4.1e-6, not of rounding size; the script prints them too. A change of sign in `l3.weight` changes the gradients of `lm.weight` and `l2.weight` by 0.45 and 0.40 of their largest entry.)

**The forms are the same arithmetic in my computation too, and my figures after one update are the worker's (D5).** I carried out the first two updates in float32 in numpy, in the order of `torch.optim.AdamW` (decay or pull, then the Adam step), with the sweep's gradient from my backward pass:

| Rule S | Mine, numpy | The worker's receipt, torch |
|---|---|---|
| decay 0.25 against pull to zero, after update 1 | 9.94e-8 (`rnn.bias_ih_l0`) | 9.94e-8 (`rnn.bias_ih_l0`) |
| decay 2.5 against pull to zero, after update 1 | 8.43e-8 (`l3.bias`) | 8.43e-8 (`l3.bias`) |
| the same two, after update 2 | 6.96e-7 and 2.09e-6 (`l2.weight`) | 1.44e-6 and 5.74e-7 (`l2.weight`) |
| two forms of the pull to the birth, after update 2, both strengths | 4.99e-8 | 4.99e-8 |

Under rule B, from its own birth, my figures are 1.37e-7 and 7.72e-8 after update 1, 1.51e-6 and 1.16e-6 after update 2, and 6.80e-8 for the two forms.

What a faulty pull would give in the same statistic after one update, rule S, strengths 0.25 and 2.5: wrong sign 5.1e-4 and 5.1e-3; the optimizer's own decay left on 2.55e-4 and 2.55e-3; no pull at all 2.55e-4 and 2.56e-3; the factor of the rate missing 0.255 and 2.56; biases left out 2.55e-4 and 2.56e-3; byte table included 2.5e-4 and 2.5e-3. A wrong sign in the second form gives 1.1e-5 and 1.1e-4 in check 3's statistic after update 2. The worker's receipts hold 9.94e-8, 8.43e-8 and 4.99e-8 at those updates.

**How many weights lie where rounding decides their sign (D5).** In update 1 of the sweep life 13,282 of the 1,118,720 weights of the six maps change sign. After it 2,975 lie within 1e-4 of zero, 36 within 1e-6, 5 within 1e-7. The two runs of check 2 (strength 0.25) differ per weight near zero by 1.8e-11 on average after update 1 and by 2.1e-10 after update 2; no weight has unequal signs in the two at either update. The density of weights at zero (1.49e7 per unit) times the mean difference gives the expected number of weights with unequal signs: 2.6e-4 at update 1 and 3.1e-3 at update 2. What it is at later updates I cannot compute without carrying out those updates. If the mean difference stayed as after update 2, the sum over updates 2 to 64 would be 0.20; if it grew as the square root of the number of updates, 0.75; if in proportion to it, 3.2. These are estimates under stated assumptions and not measurements. They say that a weight with unequal signs within 64 updates is an ordinary event, of the order of one per comparison, and they do not say more.

**One weight with unequal signs puts the next update beyond both tolerances under the sweep and not under back-propagation (D6).** From the weights after update 1 I made a copy in which the weight nearest zero of one matrix has the other sign, and computed update 2 on batch 2 for both, in float32:

| Weight with unequal signs | The two sets differ before the update by | After the update, rule B | After the update, rule S |
|---|---|---|---|
| `l2.weight[19, 283]` = ±2.7e-8 | 3.6e-7 in that tensor, 0 elsewhere | at most 3.6e-7 | `l1.weight` 1.3e-5; others at most 1.8e-6 |
| `rnn.weight_ih_l0[188, 510]` = ±2.2e-8 | 7.8e-8 | at most 4.3e-7 | `l1.weight` 2.7e-5; others at most 8.6e-6 |
| `rnn.weight_hh_l0[151, 59]` = ±1.0e-8 | 2.7e-8 | at most 5.0e-8 | `rnn.weight_ih_l0` 2.1e-3, `l1.bias` 1.3e-4, `rnn.weight_hh_l0` 1.0e-4, `l1.weight` 2.4e-5, `rnn.bias_hh_l0` 1.5e-5 |

The sizes, 1.3e-5 to 2.1e-3, are of the order of the worker's first recorded values after the rise in the tensors beyond the tolerance (6.0e-6 to 5.9e-4). In this computation the unequal sign is imposed by me; it shows what such a weight does and not that one occurred in the worker's runs.

### 3.5 Conclusion on the cause

The failure is of kind (b). Checks 2 and 3 compare, after 64 updates, two computations that agree only up to rounding, and set tolerances of 1e-5 and 1e-6. That presumes that the update is a continuous function of the weights. For rule B it is, and the checks pass. For rule S it is not: the error passed back contains the sign of each weight, 13,282 weights cross zero in the first update alone, and one weight whose sign differs between the two computations changes the next update by 1e-5 to 2e-3 of a tensor's largest entry. The four comparisons parted after updates 5 to 8, 17 to 32, and 33 to 64 (twice). Up to the parting they agree as well under the sweep as under back-propagation.

It is not kind (a): the errors a wrong pull could have are excluded under the sweep by the worker's float32 figures at updates 1 to 4, by the float64 figures over all 64 updates, by the passes of the identical file under rule B, by check 4 (the stored values equal the birth file bit for bit at updates 64 and 12,288 and every pulled tensor has moved), and by the weights of the lives themselves (section 5.2). It is not kind (c): float64 shows no large continuous growth.

What remains inferred and not observed: that at each rise in the worker's float32 runs a weight had unequal signs in the two runs. No receipt records signs, and the weights of those 64-update runs are not kept.

A consequence beyond E38, stated because it follows from the same property: under the sweep two implementations that are equal up to rounding give different weights after tens of updates, so a sweep run can be reproduced exactly only bit for bit (as check 1 does for `zero`, and E43's check 1 for `zero` and `none`), and any single figure of a sweep life carries a variation from rounding whose size E38 does not measure.

## 4. What in the E38 report depends on the failed checks

The pull is executed in four of the ten lives: `anchor` and `anchor10` under each rule. Under rule B checks 2 and 3 pass. So the statements that would be unsafe if (a) held are those that rest on `anchor` S and `anchor10` S.

| Statement of D:M/`exp45/e38/E38_REPORT.md` | If the pull were wrong under the sweep |
|---|---|
| Curves of `anchor S` and `anchor10 S` after row 0 (lines 41 and 42) | unsafe: they would not be curves of the device the brief names |
| Marks at row 12,288, `anchor` S "48.7 / 0.09 / 97.3 loses; acquires" and `anchor10` S "46.8 / 0.18 / 91.0 loses; acquires" (lines 56 and 57) | unsafe |
| Reading 5 under S: "`anchor` loses, `anchor10` loses: neither keeps, so **5**: a uniform pull is not enough on this seed. Rows 3 and 4 do not apply at either strength." (line 68) | unsafe |
| Predictions P38.3, P38.4, P38.6 "no", P38.5 "yes (91.0)", and P38.8 "no (2, 5, 6 under both)" in its part "5 under both" (lines 89 to 94) | unsafe |
| Facts, in the part that concerns the two pulled sweep lives: first row below 80 "64, in all ten lives"; wrap pairs below 50 at "64 in every non-rehearse life"; "The loss is complete within a few hundred updates in all four decay variants"; paired differences S − B for `anchor` and `anchor10`; "every sweep life stay above 94" on `min89`; "No-wrap pairs at 12,288 are 88 to 98 in every arm" (lines 60, 77 to 81) | unsafe in that part |
| Downstream, in the part "under both rules": review 9.141 "a uniform pull toward the birth weights does not protect (reading 5)"; review 9.136, addendum, "a uniform anchor is not a candidate"; the memo's lines 223, 631 and 1536; `PLAN.md` line 140 "Set aside: a pull toward earlier weights. E38 found no protection" | unsafe for the sweep, which is the rule the programme studies; safe for rule B |
| Everything under rule B: all five arms, readings 2, 5 and 6, the facts on `min89` | safe: checks 2 and 3 pass under B |
| `zero` S, `none` S, `rehearse` S: curves, marks, reading 2 ("`none` loses ... decay is not needed for the loss"), reading 6 and the cost of rehearsal, P38.1, P38.2, P38.7 | safe: no pull is made in these arms (runner lines 166 to 173). `zero` S equals E26 bit for bit (check 1), `none` has decay 0.0 read back (check 5), and E43's runner reproduced the weights of `zero` and `none` at updates 64, 256, 1,024 and 4,096 under both rules by equal hashes (D:M/`exp45/e43/checks/e43_check_1_conflict_is_e38.json`) |
| Checks 1, 4, 5, 6, 7; the item counts of `rehearse` | safe |
| Every score as a score of the saved file | safe: section 5.1 |

Since (a) does not hold, none of these is unsafe on the evidence. The table says what the qualification of section 6 covers.

## 5. E38's figures from its weight files

### 5.1 The scores (part B)

D holds 20 weight files of E38, the ten lives at rows 6,144 and 12,288 (`git ls-tree -r --name-only 7cabb13a -- MOD97_LOCAL_RULE_20261106/exp45/e38/ | grep '\.pt$'`), and the two birth files in `exp45/e26/`. The 15 other rows of each life are not committed. Each file was read through `git show` in memory, one at a time.

For each of the 20 files I recomputed ten counts with the reviewer's numpy reader (`np_gru.py`: its items, its split, its panels; the forward pass is `np_gru.forward`'s arithmetic with the maps of the input byte computed once per byte, and it agrees with `np_gru.forward` item by item on all eight panels for both births): add and sub under 89 on all 7,921 pairs without letters, split into reduction pairs (3,916) and the others (4,005); the four held-out panels of the first family at 12 letters (300 items each); add and sub under 97 held out at 12 letters (400 each).

Result: the sha256 of each file equals the receipt's `weights_sha256` in 20 of 20; all ten counts equal the receipt's in 20 of 20; the largest difference in any of the 200 counts is 0. The two birth files have the hashes the receipts record and give the receipts' row 0 (3,793 and 3,770 reduction pairs right under S; 3,838 and 3,848 under B).

The report's table of marks at row 12,288, recomputed (held-out add and sub under 89, mean / reduction pairs, mean of add and sub / held-out add and sub under 97, mean):

| Arm | S, from the weights | B, from the weights | Reduction pairs right, add + sub of 3,916 each: S; B |
|---|---|---|---|
| `zero` | 47.83 / 0.04 / 92.50 | 47.50 / 0.00 / 94.75 | 2 + 1; 0 + 0 |
| `none` | 49.00 / 0.05 / 96.25 | 44.83 / 0.00 / 96.62 | 3 + 1; 0 + 0 |
| `anchor` | 48.67 / 0.09 / 97.25 | 49.17 / 0.00 / 92.75 | 0 + 7; 0 + 0 |
| `anchor10` | 46.83 / 0.18 / 91.00 | 44.67 / 0.05 / 85.62 | 5 + 9; 0 + 4 |
| `rehearse` | 98.33 / 96.81 / 92.38 | 96.00 / 93.90 / 92.88 | 3,808 + 3,774; 3,651 + 3,703 |

These are the report's figures (lines 54 to 58). Two cells of the report are rounded up from an exact half and differ in the last digit from the same figures in its own table of curves: `none` S 96.25 is printed 96.3 in the marks and 96.2 in the curves, `anchor` S 97.25 is printed 97.3 and 97.2. The report's table of curves (110 cells of three figures) agrees with the receipts in every cell to the rounding of one decimal. Its rows other than 6,144 and 12,288 rest on the receipts alone.

The marks are far from their thresholds. "Loses" is at most 20 on the reduction pairs and the eight lives without rehearsal are at 0.00 to 0.18. "Keeps" is at least 80 and 70 and the two rehearsed lives are at 98.3 and 96.8, 96.0 and 93.9. "Acquires" is at least 85 and the lowest life is `anchor10` B at 85.62; the lowest sweep life is at 91.0.

### 5.2 The pull, seen in the weights (part C)

If the pull acted as specified, the weights of `anchor` and `anchor10` are nearer the birth than those of `none`, on each of the 12 pulled tensors and not on the byte table, by a factor that depends on the strength, and by the same factor under both rules, since the pull does not know the rule. Root-mean-square distance from the birth over the 1,122,049 pulled weights, as a share of `none`'s at the same rule and row:

| Arm | S, row 6,144 | S, row 12,288 | B, row 6,144 | B, row 12,288 | A weight that takes steps without direction and is pulled back by 0.001 k of its distance each update |
|---|---|---|---|---|---|
| `anchor` (k = 0.25) | 0.590 | 0.443 | 0.598 | 0.447 | 0.557, 0.403 |
| `anchor10` (k = 2.5) | 0.243 | 0.169 | 0.235 | 0.164 | 0.180, 0.128 |

- The shares under the sweep are within 4% of those under back-propagation in all four cells.
- Each of the 12 pulled tensors taken alone is nearer the birth than in `none`: shares 0.41 to 0.74 for `anchor` S and 0.12 to 0.47 for `anchor10` S. The byte table, which is not pulled, is farther from the birth than in `none` (1.09 for `anchor` S; 1.66 and 1.52 for `anchor10` S).
- `none` itself moves away as the square root of the number of updates (row 12,288 over row 6,144: 1.408 under S and 1.415 under B), which is why the last column is a fair comparison. The observed shares are 6% to 35% above it under both rules alike.
- The largest distance of any pulled weight from its birth value is 0.12 and 0.11 in `anchor10` S, against 0.51 and 0.83 in `none` S.

A pull that did not act would give 1.0. A pull of the wrong sign would move the weights away without bound. `anchor10` run at strength 0.25 would give `anchor`'s shares. None of these is what the files show. The comparison does not fix the strength better than to a factor of about two, because the model in the last column is crude; it does show that the two strengths differ as specified and that the sweep's lives were pulled as back-propagation's were.

## 6. Proposed ruling

E38's results under the sweep stand, with one qualification to be entered in the record: checks 2 and 3 remain failed as the brief specified them, and under the sweep the step of the pull rests instead on the worker's receipts before the two runs part (float32, updates 1 to 4: at most 1.5e-6 for check 2's pair and 1.6e-7 for check 3's) and in float64 over all 64 updates (at most 5.9e-15), on the passes of the identical file under back-propagation, on check 4, and on this note's recomputation from the birth file and from the 20 weight files; no second worker's implementation in torch has been run. The cause is the specification and not the code: under the sweep two computations that are equal up to rounding part by a finite amount at the first update at which a weight next to zero has unequal signs in the two, so agreement of weights after 64 updates in float32 cannot be required of them. No life needs to be run again; if the coordinator wants the cause observed and not inferred, the smallest run is the worker's diagnostic under the sweep (`e38_check2_diag.py S 73129 float32`, 68 CPU-seconds by its receipt) repeated by a second worker with one added count, at every update from 1 to 64, of the weights whose signs differ between the two runs, and for later briefs a check of two forms of a step under the sweep should compare them after one update, or over many updates in float64, as addendum 2 of the E31 brief already rules for two implementations of a gradient.

## 7. Commands

All from the repository root, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`, `D=7cabb13adc38753483d34dd5472fc3c84c4de675`.

| Command | What it prints | CPU |
|---|---|---|
| `python3 -I $R/e38_checks_read.py A` | section 2.3: the check receipts, the diagnostic by update, the hash of the runner in every receipt | under 1 s |
| `python3 -I $R/e38_checks_read.py B` | section 5.1: the 20 files against the receipts, the table of marks, the report's table of curves against the receipts | 103 s |
| `python3 -I $R/e38_checks_read.py C` | section 5.2: distance from the birth per tensor, arm, rule and row | 1 s |
| `python3 -I $R/e38_checks_read.py D` | section 3.4: D1 the stream, D2 the backward pass against differences, D3 and D4 the gradient under changed weights, D5 the forms, the faulty pulls and the weights near zero, D6 one weight with unequal signs | 28 s |
| `git show $D:<path> \| cat -n` | every quotation, with the line numbers given | |
| `for r in $(git for-each-ref --format='%(refname:short)' refs/remotes refs/heads); do git grep -l -i -E "check2_diag\|float64 diagnostic\|checks 2 and 3\|check 2 and 3\|checks 2, 3" $r -- '*.md'; done` | section 2.6 | |
| `git ls-tree -r --name-only $D \| grep -i -E "e38\|exp38"` | the files of E38 on D | |

The script imports `np_gru.py` and `compare_pt_tensors.py` of this folder, reads D through `git show` at the commit above, sets `sys.dont_write_bytecode`, and writes nothing. Part D carries out the arithmetic of the first two updates of a life in memory, on copies of the birth weights, and keeps nothing; I do not count that as training, and say it here so that the coordinator can judge.

## 8. What I could not check

- **Anything that needs torch.** The receipts of the checks and of the diagnostic are taken as committed. I did not rerun them. My numpy arithmetic agrees with the worker's float32 figures to three digits after update 1 and for the two forms after update 2, and to the order of magnitude and the tensor after update 2 for the other pair.
- **That a weight had unequal signs in the two runs at each rise** in the worker's float32 comparisons. This is inferred from the pattern of tensors, from the float64 runs, from the pass under rule B and from the effect of an imposed unequal sign. The run named in section 6 would observe it.
- **The rate of such weights beyond update 2.** Section 3.4 gives a range under assumptions.
- **The order of decay and step inside torch 2.15's AdamW.** I did not read its source. Check 2 under rule B and the agreement of my update 1 with the worker's support the order the brief states; the audit notes that a pull placed after the step would not be caught at 1e-5 and would not matter.
- **The 15 uncommitted rows of each life and the weights of the 64-update check runs.** The receipts hold their hashes. The scores at those rows rest on the receipts.
- **The diagnostic under rule B.** The worker ran it under the sweep only.
- **How much a sweep life's scores vary with rounding.** E38 has one run per arm and rule.
- **Other seeds.** E38 is seed 73129 only, as its brief says.
- **The files under `.py`, `.json` and `.log` of the refs after `origin/claude/vibrant-gauss-tops5f` in the first search of section 2.6** (the two last `origin/claude/` refs and the `codex` refs): for these only the `.md` files were searched, because the first search was cut off by a restart of the container.
