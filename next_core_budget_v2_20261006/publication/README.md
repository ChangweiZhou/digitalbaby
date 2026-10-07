# MiniFly next-core budget V2 — completed results

Scientific verdict: **PROMISING_BUT_UNRESOLVED**. The final audit accepted this as a completed experiment. Acceptance of the evidence does not mean adoption of a new core: the registered adoption and E3-existence gates did not pass.

The exploratory screen completed 176 jobs / 528 lives on 8 paired worlds. Its sealed choice was Q_HALF, targeting reuse relative to ERROR. Confirmation completed 192 jobs / 576 lives on 48 fresh paired worlds. Total: **368 committed whole jobs, 1,104 lives**. Q_HALF uses the ERROR learning state with its predeclared half-weight readout, so it is not counted as an additional physical learning trajectory.

In confirmation, Q_HALF had taught-cue old/new/revision accuracy of 97.66% / 99.67% / 95.31%. Never-taught choice was 61.11%, and the matched prior-teaching contrast was +8.68 percentage points; the registered interval did not establish a positive causal teaching effect. The gain in that contrast relative to ERROR was +4.86 percentage points. The harm-risk gate also did not pass. These findings are promising signals, with the adoption verdict unresolved. They do not establish general intelligence, cue-free output or learning without teacher bytes.

Read [confirmation report](../CONFIRM_REPORT.md), [exploratory screen](../SCREEN_REPORT.md), [final audit](../results/FINAL_AUDIT.json), and [summary](../results/SUMMARY.json). The protocol, source lock, scripts, qualification records and recovery audit are preserved. The filesystem repair was operational; no scientific records were rerun.

## Complete evidence archive

GitHub's file limit requires splitting the already completed RESULT_BUNDLE.zip into 15 ordered binary parts, each at most 48 MiB. These are exact consecutive bytes of the existing archive, not a replacement or reanalysis. The archive contains committed receipts, final W-state models and locked source dependencies. Recovery checkpoints and verbose job stdout are omitted by the original packaging policy.

From this directory, reconstruct it with:

```bash
cat archive/RESULT_BUNDLE.zip.part* > RESULT_BUNDLE.zip
unzip RESULT_BUNDLE.zip -d extracted_results
```

Original archive SHA-256: `a5eebdd7c885e2846dffb4e5090dea97fde8a69747d3f5baf3c008b67be44e28`. Its size and ordered part sizes are in [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json).

Historical launch records correctly say that GitHub upload was not then authorized. A later explicit user instruction authorized this publication; see [AUTHORIZATION.json](AUTHORIZATION.json). No historical scientific lock or audit was rewritten.
