I checked the live GitHub repo rather than relying on our earlier summaries. The project has moved materially since our last mechanism discussion. The current frontier is on `ChangweiZhou/digitalbaby`, branch `codex/next-core-budget-v2-20261006`, now at commit [`0b343214…`](https://github.com/ChangweiZhou/digitalbaby/commit/0b34321470924e62aead6f34f879d891a02dae63). The scientific results were published one commit earlier at [`564bbd56…`](https://github.com/ChangweiZhou/digitalbaby/commit/564bbd56cb52ca5b01e3f0e9ec89eff29f764de5). So the repo has actually **overtaken the REL2 experiment design we just drafted**.

The important files are the current [screen report](https://github.com/ChangweiZhou/digitalbaby/blob/codex/next-core-budget-v2-20261006/next_core_budget_v2_20261006/SCREEN_REPORT.md), [48-world confirmation report](https://github.com/ChangweiZhou/digitalbaby/blob/codex/next-core-budget-v2-20261006/next_core_budget_v2_20261006/CONFIRM_REPORT.md), and [frozen experiment protocol](https://github.com/ChangweiZhou/digitalbaby/blob/codex/next-core-budget-v2-20261006/next_core_budget_v2_20261006/publication/design/EXPERIMENT.md).

## Where the project is now

The older result chain remains sound. Package A established that `R3` dramatically fixes exact taught-memory E1 without solving E3. `R_center` then showed that learned-\(p\) dependence was mostly unnecessary for that E1 gain. The 64-world Latin-table experiment subsequently showed something stronger than “no generalization”: the learned influence was **actively anti-helpful** on withheld cells. The [R_center report](https://github.com/ChangweiZhou/digitalbaby/blob/study/rcenter-survivor-rounds-20261003/studies/rcenter-survivor-rounds-20261003/results/report/ROUND1_REPORT.md) has E3 \(=-8.20\) pp and only 14.45% held-out accuracy despite 97.14% old and 94.63% new taught accuracy.

`ERROR` did not fix that. The now-complete [ERROR study](https://github.com/ChangweiZhou/digitalbaby/blob/study/error-completion-20261004/studies/error-completion-20261004/FINAL_REPORT.md) gives 14.06% held-out accuracy and E3 \(=-10.94\) pp, versus 15.23% and \(-8.98\) pp for `R_center`. So corrective prediction-error writing is useful for revision but is not the missing relational mechanism.

The new budget-v2 experiment then tested exactly the families we had just re-filtered: read authority `Q_HALF`, two low-dose cue-plasticity variants, S3 cue-driven rewiring, and a three-dose multiplicative relational representation family `REL05/10/20`, with suitable controls.

That last point matters: **our proposed REL2 scale bracket has effectively already been tested in spirit.** The repo's REL implementation is not byte-for-byte identical to the markdown I drafted, but scientifically it is the same class of question: fixed multiplicative interaction representation, extra native stores, doses \(0.5,1,2\), a write-permutation control, and a first-order/additive control.

### What the screen says

The 8-world screen produced a surprisingly coherent picture:

| Configuration | old | new | revision | reuse W | reuse W−N | CPU |
|---|---:|---:|---:|---:|---:|---:|
| `ERROR` | 95.83% | 99.61% | 96.88% | 68.75% | 12.50% | 82.5 s |
| `Q_HALF` | same learning | same | same | **79.17%** | **22.92%** | 82.5 s |
| `S3_CUE` | 95.83% | 99.61% | 98.44% | 70.83% | 16.67% | 115.9 s |
| `REL05` | 94.79% | 99.22% | 90.62% | 70.83% | 18.75% | 144.9 s |
| `REL10` | 88.02% | 95.31% | 87.50% | 72.92% | 20.83% | 144.5 s |
| `REL20` | 82.81% | 93.75% | 82.81% | 70.83% | 18.75% | 143.3 s |

The REL result is informative even though it did not advance. The three doses all moved the simple reuse endpoint in the desired direction, but the price was clear and dose-dependent: exact-memory/revision damage plus roughly **1.75× online CPU**. `REL05` already failed the parent revision-protection and CPU gates; increasing the dose worsened the E1 tradeoff. The exact current four-extra-store multiplicative REL implementation therefore should **not** receive another \(0.25,0.75,1.5\) tuning campaign.

There is nevertheless a weak mechanistic hint: `REL10` had reuse \(W-N=20.83\%\), while the write-permuted control was 12.5% and the first-order control 16.67%. But \(n=8\) was explicitly exploratory, so this is not a mechanism claim.

S3 is similar. It looks healthy—excellent E1, slight reuse improvement, CPU only ~1.40×—but it did not clear the frozen practical-gain criterion and therefore was not confirmed. It remains a reserve candidate, not a positive result.

## The genuinely new result is `Q_HALF`

The selection procedure chose `Q_HALF`, which does **no new learning at all**. It reuses the `ERROR` trajectory and simply halves private readout authority.

That candidate then received 48 fresh confirmation worlds. The result is much more interesting than its formal verdict `PROMISING_BUT_UNRESOLVED` sounds:

\[
\begin{array}{c|ccc}
 & CENTER & ERROR & Q_{\frac12}\\
\hline
\text{old E1} &94.97&97.74&97.66\\
\text{new E1} &95.64&99.74&99.67\\
\text{revision E1}&37.76&96.35&95.31\\
\text{reuse raw}&54.86&57.64&\mathbf{61.11}\\
\text{reuse }W-N&0.69&3.82&\mathbf{8.68}
\end{array}
\]

So `Q_HALF` improved raw held-out reuse over `ERROR` by **+4.86 pp**, with its registered one-sided lower bound still positive at the adoption alpha: +1.84 pp. Its efficiency gain was also convincingly positive because it is almost computationally free.

But it failed adoption for two separate reasons.

First, the causal prior-teaching effect itself was not quite established under the stricter E3 claim:

\[
W-N = 8.68\text{ pp},\qquad
LB_{\alpha=.01}=-2.34\text{ pp}.
\]

Raw held-out performance is convincingly above 50%, but we cannot yet establish that the excess is causally attributable to old teaching at the registered E3 confidence level.

Second, 4 of the 48 worlds had at least one \(>5\) pp lifetime degradation relative to `ERROR`. The resulting one-sided upper confidence bound on the severe-harm probability is about **19.5%**, above the prespecified 10% ceiling.

That second failure should not be dismissed as mere underpowering. If the observed 4/48 rate remained stable, the harm-probability gate would require an enormous sample before its 97% upper bound fell below 10%. By contrast, the E3 lower bound by itself would probably resolve with a much more modest sample if the present mean and variance replicated. In other words, **the main problem with global `Q_HALF` is not simply n**.

## This changes the mechanism diagnosis again

We now have a very useful apparent contradiction:

\[
\boxed{
\alpha_{\rm private}=0.5
\text{ helps the simple binary reuse task}
}
\]

but

\[
\boxed{
\alpha_{\rm private}=0.5
\text{ clearly hurts the Latin completion task.}
}
\]

That is exactly what we needed to learn.

It means the old read-authority family should **not** be declared dead, but neither should we adopt a global scalar mixture. The correct conclusion is:

\[
\boxed{
\text{private-memory authority has to be query/evidence dependent.}
}
\]

This brings the old V37/CORR2 lesson back into focus, in a cleaner form. Private episodic memory is extremely useful when the query is a directly learned association. It can become actively misleading when a query should be answered from shared relational structure. But blindly suppressing it is also wrong, because the shared route itself is not universally trustworthy—the Latin shared-only result was worse than the combined readout.

So the next missing computation is no longer simply “representation” versus “readout.” It is closer to

\[
\boxed{
\text{earned authority:
when should shared structure override episodic evidence?}
}
\]

## What I would do next

I would **not run the REL2 markdown we just produced as written**. The new GitHub work has already supplied enough evidence to supersede it.

The next round should be much cheaper and more diagnostic:

1. **Exploratory read-only localization on existing receipts.** Use the already saved shared/private score vectors from the 48-world `ERROR/Q_HALF` confirmation and from the Latin `R_center/ERROR` studies. For each probe record only legal pre-answer observables: shared top-1/top-2 margin, private margin, shared/private agreement, margin ratio/difference, perhaps existing native support quantities if they genuinely exist. Ask whether the cases where `Q_HALF` helps versus hurts are separable by any of these observables. This is post-hoc and earns no efficacy claim.

2. **Freeze exactly one earned-authority rule.** For example, a rule of the form
   \[
   \alpha_p(x)\in\{1,\tfrac12\}
   \]
   chosen from a local confidence/disagreement criterion, with no task ID, no Latin-row information, no future answer and no learned evaluator labels. The exact rule should come from the diagnostic, then be frozen. Do not open a threshold tournament.

3. **Freshly test that one rule on both E3 families simultaneously.** This is crucial. A rule that wins only on the simple binary-relation task but reproduces the Latin failure is not a core improvement. Because this is read-only arbitration, `ERROR`, fixed `Q_HALF`, and the new `Q_EARNED` can all be scored from the **same learning trajectories**, so the scientific cost is modest. Require positive causal reuse and above-chance raw performance in both families, plus E1 preservation.

4. **Keep `S3_CUE` as the next fallback**, not REL. S3 passed the screen's engineering guards, maintained excellent exact memory, and has a plausible reuse signal at only ~40% CPU overhead. It did not meet the frozen advancement delta, so it needs a genuinely fresh protocol before any positive claim. If earned read authority fails, S3 is the next family I would spend compute on.

5. **Park the current REL implementation.** The screen did what we wanted the scale bracket to do: it showed a reproducible-looking reuse/E1 tradeoff across \(0.5,1,2\), and the compute cost is largely structural rather than dose-dependent. A future REL mechanism would need to be qualitatively different—sparse/conditional interaction computation, substantially fewer extra stores, or an interaction representation whose authority is earned at read time. Simply reducing the dose again is not justified.

So the project has narrowed further. The current best *learning rule* is `ERROR`; the current strongest *new mechanistic clue* is `Q_HALF`; the current REL architecture is too costly/destructive; and the immediate question is now:

\[
\boxed{
\text{Can MiniFly learn when episodic/private evidence should yield
to shared relational evidence?}
}
\]

That is where I would put the next experiment. The GitHub work makes this a better-supported direction than the REL2 experiment I proposed before seeing the latest branch.