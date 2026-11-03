# V184 design-only decision record — 2026-11-02

Anchor: \`ChangweiZhou/digitalbaby\` \`codex/minifly-consolidated-20261023@840f2a7b9dcd25f4262c3e17058bb1cf30139bf2\`, with V183-R1 publication \`840f2a7b9dcd25f4262c3e17058bb1cf30139bf2\`.

Question: Are conditional H64 features present but inadequately deployed by T64→O257?

Falsifier: Retain T, add fixed \`0.5 * H\` at its existing output-feature port with zero new learned parameters; compare the full continuing learner against NORMAL and an actual post-dropout feature-norm sham. Beat each on held A2 with record-level CIs, its own mean forecast and the unigram; require within-checkpoint H alignment and later-life B/correction/EOS guards.

What ran: Source/receipts and integrated skill inspection only; **0 V184 optimizer-record applications, 0 V184 scored scientific lives, 0 qualification attempts**.

Last outcome: V183 HPC 4.587709 bits vs NORMAL 4.582233, SHUFFLED 4.615474 on M1 A2 held; HPC did not beat mean forecast/unigram and did not generate own EOS in recorded 288 continuations. This does **not** establish what H64 can or cannot carry.

Lesson: Another H credit tweak is not presently justified; simple removal of T was worse in V182. A simultaneous *retained-T* H-output route tests a new, relevant architectural choice at unchanged capacity and training exposure.

Disposition: **PROPOSE V184-HRES-D1 as DESIGN_ONLY**. Qualification and model development remain owner-authorized separate actions. Formal MRK-1 checker unavailable; no independent cross-family certification, sealed confirmation, E1/E3 or adoption.
