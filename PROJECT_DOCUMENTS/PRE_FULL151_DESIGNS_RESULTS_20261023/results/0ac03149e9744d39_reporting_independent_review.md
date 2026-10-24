# Independent review of V43 reporting qualification

Reviewed the reporting module against the locked protocol and final operational gate definitions. Read-only review; reporting implementation remains owned by the reporting agent.

## Concrete fixes requested before source freeze

1. **Q7 missing endpoints must not imply earned complexity.** In the reviewed implementation each simpler policy's `matches` becomes False if any needed pair count is below16. `not any(matches)` then turns missing comparisons into Q7 pass. Require all three endpoints × both primary loads × both simpler policies to have16 finite pairs before evaluating Q7; otherwise return not_evaluable. This is an actual fail-open gate, even if current engine outputs normally populate every metric.
2. **Q10 needs complete whole-loss pairs.** Require16 finite paired whole-loss values at both128 and256, not merely non-null means. The relative held-out change already requires16.
3. **Omnibus interaction test must not silently change hypothesis.** The reviewed model computes a Wald/F statistic through the covariance pseudoinverse and uses its rank as numerator degrees of freedom. With rank-deficient interaction covariance this can ignore nonzero coefficient components in the covariance nullspace. Prefer marking the omnibus F/p unevaluable unless the full planned interaction block has full rank, while retaining individual exploratory coefficient estimates and intervals. An explicitly prespecified estimable subspace would be an alternative, but must be labeled.
4. **Q0 assertion coverage should match the promised event count.** A positive firewall-check count establishes that at least one check ran, not that every byte was checked. Require firewall_checks==byte_events, with separate nonzero probe checks as appropriate. Root's final engineering-cache envelope and reporting's reader must agree.
5. **Q0 wording.** The gate description says mature acquisition, whereas the adopted quantity is unchanged SLOTS accuracy on every intended current-wave identity at learned aliases0/1, at all four checkpoints. Change the description to avoid implying survivor-conditioned mature-only denominators.

## Reviewed gate logic that matches the adopted protocol

* Q0 acquisition checks all four named wave checkpoints and nonzero population denominators. Required128/256 loads use the exact sixteen-seed roster; incomplete or invalid cells do not qualify.
* Q1 separately requires500G events in every seed at both primary loads and seed-equal mean oracle headroom≥0.20.
* Q2 averages32 seed/load anchor means equally and caps every individual seed/load disadvantage at0.05.
* Q3 is checked separately at128 and256; shared physical learning supplies the consolidation invariant.
* Q4/Q5/Q6 preserve the fixed effect-size margins and paired95% bounds. Q6 uses the matched permutable subset and requires80% coverage per seed.
* Q7 signs are correct for LOCAL−simple differences: a better simple policy counts as matching. It uses point-estimate practical margins rather than claiming statistical equivalence.
* Q8 uses the average of seed-level capture ratios and rejects undefined denominators. Ratios above1 are retained.
* Q9 retains the specified whole-loss tolerance plus Q5.
* Q10's relative held-out degradation difference has the correct sign and uses the adopted2pp margin.
* OutcomeA excludes Q10 exactly as the original decision formula does, while reporting scaling status separately.

## Statistical unit and multiplicity

The paired estimates operate on seed-level differences; authority events are not passed to Student-t inference as replicates. Duplicate seed/load records are rejected. Holm correction uses the four mechanistic comparisons separately at each load, with load256 identified as the primary family and the other loads exploratory. Missing tests do not shrink the family size. SUPER remains outside that four-test family.

The load model and bootstrap restrict to loads with all16 physically valid seed cells. Bootstrap resampling preserves each seed's complete vector of policies and loads. The held-out model is transparently a linear probability contrast model, not a calibrated extrapolator. L50 is limited to a unique crossing within the observed load range; multiple crossings and no crossing return NA. Conditional bootstrap L50 intervals include their identifiable-resample fraction.

Partial/invalid-load tables presently still show valid-survivor paired estimates and p-values alongside flags. If retained, these must be labeled descriptive/exploratory in the rendered primary-contrast table too; they cannot be interpreted as the completed preregistered sixteen-seed comparison. This is a presentation boundary, separate from the correctly conservative qualification roster check.

## Remaining verification scope

The reporting agent is checking integration against actual smoke schemas. This review did not edit the module or execute scientific data. After the identified fixes, synthetic missing-endpoint tests should exercise Q7 and Q10, and a rank-deficient interaction fixture should verify no silent reduced-hypothesis p-value is emitted.
