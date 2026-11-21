# Errata and closure for the outside-advice folder (21 November 2026)

Written by session `session_017dSEpWqUVkyw5SyyeE6M13` on branch `claude/magical-albattani-sei232`, after merging the hub `claude/friendly-darwin-8mmhfm` at `f31b2dee5` and bringing in `nano/ref2/` from `claude/nano-ref2-8mmhfm` at `fb887a2f2`. Append-only: no earlier file in this folder is edited by it; where it and an earlier wording differ, this note is the later record. Development level; no checker gate, seal or outside review is claimed. No run was made to write it.

## 1. The reference network's figure 1.454

**Correction.** The figure 1.454 used for nanoGPT's full network in `ROADMAP_SECOND_REVISION_20261120.md` (ledger row L0 and elsewhere) was measured on the every-eighth subset of the last tenth, which is answered at a block's last place. On all bytes the four-seed mean is **1.4590 nats a byte (sd 0.0011)**; on the every-eighth set the four seeds give 1.4653 (sd 0.0078), running from 1.457 to 1.473, and the earlier 1.4541 is below all four.

**Source.** `nano/ref2/REPORT.md` (its table and reading 2), from `nano/ref2/PLAN_1.md` entry 3 (commit `9b757f636`, 21 November 02:10 UTC, on `claude/nano-ref2-8mmhfm`).

**Where 1.454 appears in the roadmap** (`grep -n "1\.454"`; the lines are left as written): 58 (the reference table), 74, 82, 91 (row L0), 187 (twice), 312 (twice), 397, 437 (as 1.4541), 710 (section 30, item 10: "1.454 to 1.456, one shortened run"). Line 372 also matches, but its 1.4541 is a figure of the system with word cells, not of the network. `STATE_AT_THE_END_OF_20_NOVEMBER.md` has no occurrence.

What this changes in comparisons is set out in `nano/ref2/REPORT.md` itself (readings 4 and 5); it is not restated here.

## 2. Plans with no outcome

`nano/lives/PLAN_12.md`, `PLAN_15.md` and `PLAN_16.md` have no outcome. The session that wrote them (`session_017dSEpVed7SthEH3ngGAdNc`) stopped at a usage limit at 03:47 UTC on 21 November; its last commit is `f31b2dee5` (03:37 UTC). The figures it printed after that commit, for `PLAN_16.md`'s development runs and part of `PLAN_15.md`'s reported runs, are recovered verbatim, unverified, in `nano/lives/RECOVERED_FROM_TRANSCRIPT_20261121.md`, which also lists the runs that were unfinished. The plans are not edited; none of their predictions is scored here.

## 3. The counting and compression line closed as a candidate for the learner

**Ruling.** Mr. Xu, 21 November, as recorded in `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/WHAT_A_NEURON_HAS_20261118.md`, section 20, answering three cases ((a) a part of the learner reading a long window of past bytes; (b) a small store of episodes used for rehearsal; (c) the whole history kept for retrieval):

> （b)是可以的，(a)确实不可以，也不是持续学习而成为信息压缩了。

Translation (that section's): "(b) is allowed. (a) is indeed not allowed; that would no longer be persistent learning but information compression." Section 20 takes (c) as excluded with (a) (its own reading, not his words) and states: "Systems that keep the whole history, such as context-mixing compressors or a full-history match model, are not candidates for the learner."

**Closure.** On that ruling, the line of systems of counting cells measured as compressors in this folder (the counting systems of `nano/table`, `nano/suite` and related lines, and their comparisons with xz, bzip2, gzip and the context-mixing family) is closed as a candidate for the learner. Its results are kept as written, as historical evidence, and may be cited only as a yardstick. Section 20 itself notes that whether such systems may still be measured beside the learner as a yardstick in future is not settled by the ruling. A small store of episodes for rehearsal (as in `nano/lives/PLAN_15.md`) is allowed by the same ruling.

## 4. Open, for the owner

Whether numbers shared by all cells of a kind ("shared trusts") count as local learning is open and not decided. Per roadmap section 29, the shared trust came in at ledger row L-C and every counting system from L-C onward that reads through an answering cell with trusts rests on it; Mr. Xu's words of 21 November were "我们反复强调是local learning" ("we have stressed again and again that it is local learning"). Section 20 of `WHAT_A_NEURON_HAS_20261118.md` lists the same question as open.

## 5. Section 30 of the roadmap

The wording corrections in section 30 of `ROADMAP_SECOND_REVISION_20261120.md` stand as written there. The figure in its item 10 ("1.454 to 1.456, one shortened run") is subject to item 1 above.
