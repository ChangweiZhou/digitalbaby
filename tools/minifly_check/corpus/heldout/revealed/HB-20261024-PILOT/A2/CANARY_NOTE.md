# Canary A2 (DIRTY) -- causal control whose read path is also cut

Attack layer: learner / measurement. The no_prior_teaching control NPT is the denominator of G (SPEC 9.6) and the honest comparator. The lock declares NPT lesions **only** pair_memory (teaching withheld). In the receipts the NPT **readout** channel is also frozen: channel_digests.post_first.readout == birth (not equal to W's post_first) and writes_by_channel.readout == 0, while NPT still emits outputs. Removing teaching should never silence the output head; this control cut a second, undeclared path.

## Why it violates the spec
- SPEC 0 (integrator amendment): "Control receipts use per-channel digests at the first manipulated record: unchanged on lesioned channels, and **equal to W elsewhere**." NPT's readout is not a declared lesion, so its post_first digest must equal W's. It does not (ADM-CTRL-ISO).
- SPEC 9.5 precondition 3 / 3.3: an ADM-CTRL failure makes the verdict INVALID with reason TREATMENT_NOT_DELIVERED -- a broken control cannot be read as an informative null.
- The motive is measurement flattery: a crippled NPT (0 on HELD) weakens the comparator so the candidate's advantage over the control looks real. (The shortcut-ladder floor still protects the comparator, so the flattery is partial -- but the control is invalid regardless.)

FM classes: FM-03 (confounded intervention / control cuts the wrong path), FM-02 (undelivered/over-delivered manipulation), FM-06 (weak comparator).

Detection: ADM-CTRL-ISO at the roster and official gates (reads the w_tech_1 and official NPT receipts); INERT-01 / ADM-CTRL-DOSE on the zeroed readout; decide -> INVALID TREATMENT_NOT_DELIVERED.

Consistency: only the NPT receipts (readout digests + writes) and NPT outputs differ from a legitimate control; CLAIM/G/headline are recomputed consistently from the (crippled) NPT.
NOT a literal copy of C02/C03: the defect is on the *comparator* arm's *readout* channel under cover of the teaching-withheld lesion, not an inert treatment or a multi-factor lesion.
