**The new work changes the next priority: a larger decoder is not the answer. We now have evidence of two separate failures—an external decoder that damages usable native outputs, and native NEW writes that destroy access to OLD content. These should be repaired separately.**

I reviewed **`8eb6ec57`**, published October 22, 2026, at 11:00 p.m. Eastern, including the completed RECOVERABILITY V1 pilot, its source, saved diagnoses, artifact audit and proposed revision. Unlike the earlier partial exercises, this pilot completed its declared roster. **All four tested configurations failed the joint OLD-and-NEW recovery criterion.** No Transformer host or new arithmetic experiment was run.  

My recommendation is **a small stopping-only readout test using existing training artifacts, alongside one focused native anti-interference experiment—not another large decoder campaign before addressing storage.**

## 1. What the new experiment has established

The experiment compared two storage configurations—ordinary centered addresses, **RAW**, and standardized addresses, **STD**—with two ways of exposing native responses to the decoder: averaging two output components, **MEAN**, or preserving them separately, **SEP**. The external readout had 5,801 parameters and was frozen before fresh evaluation values arrived. 

Final complete recall of **nonempty values, including the model’s own EOS**, was:

| Configuration | OLD after NEW learning and rest | NEW after rest |
|---|---:|---:|
| RAW–MEAN | 0.26% | 0.52% |
| RAW–SEP | 0.26% | 1.56% |
| STD–MEAN | 0.26% | 3.91% |
| STD–SEP | 0.00% | 3.91% |

These are descriptive averages over four assessment pairs, two value lives per pair and three readout initializations. The readouts and paired lives are correlated observations, not additional independent experiments. **Standardization helps some NEW recovery, but neither standardization nor exposing separate output components establishes retained memory.** 

That is the negative result. The more useful result is what the diagnoses reveal underneath it.

### Native memory often contains a correct recently learned payload

When the evaluator temporarily ignores stopping and checks the payload over its actual length:

| STD memory state | Direct native payload exact | Learned MEAN decoder payload exact |
|---|---:|---:|
| OLD immediately after teaching | **54.69%** | 14.58% |
| OLD after NEW writes and final rest | **0.00%** | 0.78% |
| NEW after final rest | **61.72%** | 20.57% |
| OLD at the matched final time, with NEW writes disabled | **53.91%** | 9.90% |

These are **diagnostic payload scores**, not autonomous complete-recall successes. Nevertheless, they establish something important: the native system is substantially better at preserving recently acquired payload content than the whole-output score alone suggests. 

Two different problems are visible:

**The decoder problem:** it changes many payload bits that were already correct.

**The storage/access problem:** substantial NEW writing changes OLD native responses so much that none of the evaluated OLD nonempty payloads remains completely correct under direct decoding.

A single intervention should not be expected to solve both.

## 2. My previous decoder recommendation has now been tested—and needs narrowing

I previously recommended a small decoder because strong analog memory signals might contain information that direct thresholding missed. That was a reasonable experiment. **This particular decoder did not deliver the proposed benefit.**

Its architecture included a direct-output skip connection, but that did not protect correct payload signs. The learned residual was allowed to change all eight payload logits as well as EOS. Its shared hidden layer was trained on both payload and stopping losses. 

The result does **not** prove that every learned decoder will fail, or that optimization of the stopping objective alone caused the payload damage. But it does show why the next version should have a stronger functional boundary:

> **A component added to improve stopping should not be allowed to rewrite the content that native memory already gets right.**

This is a functional design principle, not an argument for more hidden layers.

### Exposing additional native output components was not enough

Separating the two native output components did not produce a joint OLD-and-NEW advantage. At STD, both observation types ended at 3.91% complete NEW recall, with essentially no OLD recovery. 

I would therefore keep **MEAN as the simpler primary interface** for the next bounded readout test. The separate components remain available for diagnosis, but they do not presently justify another architecture search.

## 3. The stopping-only revision is sensible—but it cannot be the whole next strategy

The repository proposes **STOP-ONLY**: payload bytes come directly from native signs, while an external learner can modify only the EOS decision. It retains the same native learning rule and gradual schedule. This proposal is unexecuted. 

**I agree with the intervention. I disagree with treating it as something that must solve joint OLD-and-NEW recovery before native protection work begins.**

There is a simple mathematical reason. For a readout that cannot change payload bits,

\[
\Pr(\text{complete correct value and EOS})
\leq
\Pr(\text{direct native payload is completely correct}).
\]

On the saved STD final-OLD states, the right-hand side is **zero**. Therefore, on those states, **no stopping-only learner can produce positive complete OLD recall**, irrespective of how well it learns EOS.

Fresh worlds may differ, but the existing evidence gives no reason to expect a stopping repair to eliminate the OLD-content problem.

This does not justify weakening the final system objective. It means distinguishing two experimental claims:

**Stopping claim:** can the new module convert already-correct native payloads into complete, correctly terminated answers?

**Persistent-memory claim:** can the system preserve those payloads while acquiring NEW content?

The final system must satisfy both. A module experiment should be evaluated on the function it can actually change.

### There is a real stopping opportunity

The learned head improved EOS ordering considerably. For STD final NEW, the true EOS score exceeded every earlier stopping score in **99.22%** of nonempty queries, compared with **88.28%** for the direct response. Yet its actual fixed-zero-threshold stopping decision was valid for only **21.88%**. 

That suggests that **relative stopping evidence is often present, but the usable decision rule is poorly calibrated**.

However, 99.22% is not stopping accuracy: it uses the true length to assess whether an item-specific separating threshold exists. Such a threshold cannot be supplied during actual recall.

The posthoc recombination—native payload signs plus the existing frozen head’s EOS—raises STD final NEW complete recall to approximately **13.3–13.8%**, while OLD remains zero. This was not a preregistered candidate, so it is a lead for the next test rather than a replacement for the negative primary result. 

**Practical conclusion:** test STOP-ONLY, but keep it small and do not claim that it addresses native interference.

## 4. The main native research priority is now selective write protection

The strongest causal observation is the contrast between OLD after NEW writing and OLD in the matched write-disabled continuation:

\[
\text{OLD payload exact: }0\%
\quad\text{versus}\quad
53.91\%.
\]

The matched continuation still experiences time, NEW inputs and passive dynamics. What differs is NEW value writing. This supports a substantial write-dependent loss of OLD recoverability under the tested interface. It does **not** prove that every trace of the information has been irreversibly erased from the full native state. 

This is the point at which the older **neuron availability, local protection and stable/flexible allocation** ideas become relevant again. Their relevance now comes from a measured problem, not from their biological names.

### The functionality to add

I would prioritize:

> **A local mechanism that reduces destructive changes to pathways already supporting stable, useful associations, while leaving enough plastic capacity for NEW learning.**

That is different from globally lowering the learning rate, freezing the entire memory, or giving OLD and NEW separate task-labelled banks.

A candidate could maintain a bounded, continuously updated measure of a connection’s or cell’s contribution to successfully learned responses, using only actually observed teaching events. New updates would be attenuated selectively according to that measure. The precise rule needs to be specified before testing; “importance” cannot remain an undefined label.

There are two essential comparison conditions:

| Comparison | What it rules out |
|---|---|
| The same total update attenuation, applied uniformly | An apparent protection gain caused only by learning less |
| The same distribution of protection strengths, assigned to the wrong cells or connections | A gain unrelated to which pathways actually matter |

Both OLD and NEW must be measured. **A protection mechanism that preserves OLD by preventing NEW acquisition has not solved the problem.**

This should be one explicit candidate, not a simultaneous combination of protection, replay, new time constants and new routing. Some related historical recipes already failed; a proposed version must state what operation differs rather than reopen them by renaming them.

### One qualification: the current schedule gives NEW stronger nominal writes

Both cohorts receive fourteen passes, but they do not receive the same nominal teaching multiplier. The gradual schedule continues across the life, so NEW occurs later on the ramp and receives stronger record-level coefficients. That is visible in the implementation. 

This does not invalidate the matched NEW-write-off result: NEW writes genuinely cause damage in this protocol.

It does mean we should not yet conclude that the anatomy has exhausted its memory capacity or that equal quantities of OLD and NEW learning are being traded off. Record-level strength, actual residual, elapsed time and useful acquisition are distinct quantities.

Keep the authorized gradual schedule. But instrument the proposed protection comparison so that it measures **actual useful NEW learning versus OLD damage**, not merely nominal pass counts.

## 5. How I would organize the next work

I would use two small, separately interpretable experiments rather than a long sequence of mandatory qualification rounds.

### A. Stopping-only readout: reuse the available training artifacts

Implement the proposed invariant: **head updates may change EOS, never payload signs**.

Fit on the already designated training episodes. Previously inspected development and assessment traces may be used for exploratory diagnosis, but they are no longer fresh evidence. Freeze all choices before generating a small fresh evaluation set. The repository’s revision already preserves this distinction. 

Evaluate complete recall, while also reporting stopping success **conditional on the native payload being correct**. Include OLD immediately after acquisition, NEW after acquisition, and the matched passive OLD continuation. These states contain a measurable supply of correct payloads, so they can reveal whether stopping improves.

This conditional metric is a diagnostic, not permission to choose favorable queries at deployment or to replace the eventual whole-system endpoint.

If STOP-ONLY cannot improve complete recovery on these usable states, stop expanding this readout family. Do not move automatically to a larger neural network.

### B. Native protection: hold the readout fixed

Test one local write-protection operation against the unmodified normalized native memory and its matched controls.

The first decisive question is whether **OLD payload exact** improves without destroying **NEW payload exact**, under the same information stream and resource accounting. Then score complete answers through the same frozen stopping interface across all arms.

Do not jointly retune the stopping head for whichever storage arm looks best. Otherwise an apparent storage improvement may be a readout-calibration difference.

The resulting decisions are straightforward:

| Result | What to do |
|---|---|
| Stopping improves, but OLD content still collapses | Retain the stopping module; continue the native protection question |
| Protection preserves OLD while NEW remains strong | Test the combined system on fresh episodes |
| OLD improves only because NEW learning weakens | Reject the claimed retention–plasticity advance |
| Both interventions fail in their assigned functions | Reconsider the representation/write mechanism, not decoder size |

This arrangement lets the project make progress on both diagnosed failures without conflating them.

## 6. What should remain parked

**A full Transformer host should remain deferred.** The learned readout pilot has not established reliable binding: no changed nonempty key in the counterfactual-life test was decoded correctly in both lives by any tested head seed/configuration at either final cohort. The external host would currently inherit an unreliable memory channel. 

**Another increase in native cell count is not yet justified.** The experiment establishes write-induced interference, but it does not isolate a simple population-capacity limit.

**The arithmetic track should remain separate.** These are directly taught random associations at surrogate addresses. Even a successful stopping/protection combination would establish a better memory component, not discovery of addition rules. The new work explicitly leaves the arithmetic track unchanged. 

**Do not reopen the entire 200-plus mechanism inventory.** The present results activate a much narrower set: useful-output preservation, selective write availability and interference control.

## Overall judgment

The pilot is a negative capability result, but it is not an uninformative one. It has narrowed the next design choices:

**The standardized native system often stores the recently taught payload correctly.**

**The joint decoder frequently makes that content worse while trying to learn stopping.**

**Substantial NEW writes separately destroy OLD recoverability.**

Those findings support a more disciplined architecture:

\[
\text{native content storage}
\;+\;
\text{restricted stopping module}
\;+\;
\text{selective protection during new writes}.
\]

Each addition has a specific function and can be independently tested. None needs to be described vaguely as “another hidden layer.”

**My main change to the repository’s proposed next step is therefore one of priority: run STOP-ONLY as a limited component test, but move native interference protection forward now. Do not spend another full pilot asking an EOS-only repair to recover OLD payloads it is forbidden to change.**

The next meaningful milestone is **simultaneous complete recovery of relevant OLD and newly acquired values**, with causal controls showing that the native memory—not an auxiliary cache or the decoder’s training history—supplies the content. Only after that should this external-memory line advance toward a larger host.