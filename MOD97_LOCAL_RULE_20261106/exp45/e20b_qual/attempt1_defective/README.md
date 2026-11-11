# Attempt 1 of the E20-B stage-1 qualification runs: defective, kept as written

Runner at commit 7a7a60a6. Births G0 to G3 and lives G0 (from its E20 birth), G0 parity (from the reference birth), G1 and G2 ran to completion
at E11_DEV_UPDATES=2048 E11_STAGE_UPDATES=256, seed 73129, one thread. The G3 life was stopped by the worker during F1.2 (before
it completed), once the defect was found.

**Defect.** `exp20_reference.life` appended each stage's record (stages 1 to 6) without its `scores` field (`scores=sc` missing); stage 0's
scores were recorded. The per-stage scores exist only in the logs, which print a subset. So these receipts cannot show parity panel by panel,
and they were not used for any check. The logs' printed subset (F1 at k = 0, 8, 12, 20) agreed between the reference life and the parity life at every stage.

**Fix.** One line, `scores=sc` added to the stage record. Because every E20 birth binds the source hashes of the runner and a life refuses a
birth made by other sources (amendment 12), all four births were redone with the fixed runner, as were the lives. The reference birth and life
(exp16_integrated.py, unchanged) were not rerun; they are in ../reference/.
