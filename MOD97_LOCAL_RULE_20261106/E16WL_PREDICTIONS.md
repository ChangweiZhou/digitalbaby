# E16-WL predictions

Frozen before qualification and training. Six 57,344-update lives use the E8 blocked schedule, window 18, width 512, batch 64, learning rate 0.001, and standing seeds 73129, 93683 and 116089. Arm C retains the E16-WR2 observed replay sampler and Store, with decay 0.25. Arm D uses E8's standard backprop setup, decay 0.3, joint sampling of observed training examples, and no Store. Comparisons of the loss use each arm's own reference. A difference between C and D also includes their data and Store policies.

1. C all-byte loss yields a mean below 60% across the eight operations after stage eight, with add97 and sub97 each below 20%, in every seed. Compare against WR2 Part B.
2. D all-byte loss yields at least 85% mean at stage eight in every seed. Record whether it falls below 60%, the brief's failure threshold. Compare against the archived E8 D arm.
3. Final mean held-composition accuracy falls by at least 15 percentage points for C and by at most 10 points for D, against their respective answer-loss references.
4. Report token-weighted prompt and answer cross-entropy in nats separately, across each entire training stage. Answer tokens include EOS. Padding contributes to neither loss nor Store writes. C retains the existing schedule for when Store writes occur.

WR2 Part B took about 1,180 CPU seconds per life and 23 minutes wall time with nine concurrent jobs. Six lives at that pace would take about 2 CPU-hours. Scoring every real byte and additional C Store writes add work. Budget 2–4 CPU-hours and roughly 25–45 minutes after the machine is free, plus short qualification. Record measured CPU, wall time and peak RSS.
