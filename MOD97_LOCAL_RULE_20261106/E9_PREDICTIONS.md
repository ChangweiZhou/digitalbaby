# E9 predictions and execution lock

Source branch codex/minifly-consolidated-20261023 at 2fc1ef2. Authorized by Mr. Zhou's request to execute the E9 brief. Read the task at that commit. Use the existing width-512 byte-window learner, splits, store, loss, optimizer, scoring checkpoints and seeds 73129, 93683, 116089.

## Predictions written before launch

A. At replay share 1/4, the mean of eight operations after their eighth stage is within five percentage points of share 1/2, with less erosion of carry-type operations after composition. Share 1/8 loses over 15 points on the earliest operations. Report each operation, each held composition pair, and every seed. The first four tasks define earliest operations. Carry-type erosion is the mean add97, sub97, add73, sub73 stage-8 minus final score.

B. Interleaving without replay holds every operation at least as well as blocked replay after learning the operations, and loses less than ten points on the carry-type mean after composition. The interleaved stage contains 49,152 updates, with eight samples per task in each full batch (four new samples per task after selecting a half-batch replay share). Interleaved replay samples only previously delivered prompts, stored up to 1,024 distinct prompts per task, under the same exact-key store and model-generated-answer rule.

C. Plain scratchpad with sign feedback, hidden 512 and 16,384 updates, gets held-pair mean at least 70% in every seed. Report the last versus previous checkpoint to show whether performance is still rising.

## Exact commands

Use the same Python environment for all runs. In this checkout it is /Users/pencilbard/Documents/second_try/.venv/bin/python (Python 3.11.5, Torch 2.14.0, NumPy 2.4.6). Commands are run from MOD97_LOCAL_RULE_20261106.

    python exp8_life.py C 73129

This default C run is the full-run profile and baseline replay check. After it completes, record its actual CPU and wall time and forecast the complete campaign before launching the remaining configurations.

For every seed S in 73129 93683 116089:

    E8_REPLAY_SHARE=1/8 python exp8_life.py C S
    E8_REPLAY_SHARE=1/4 python exp8_life.py C S
    E8_REPLAY_SHARE=1/2 python exp8_life.py C S
    E8_REPLAY_SHARE=0 python exp8_life.py A S
    E8_SCHEDULE=interleaved E8_REPLAY_SHARE=1/2 python exp8_life.py C S
    E8_SCHEDULE=interleaved E8_REPLAY_SHARE=0 python exp8_life.py A S
    EXP45_HIDDEN=512 E7_DOSE=16384 python exp7_decon.py SCR S

Reuse archived share-1/2 C, A and backprop ceiling receipts only after default-path parity is verified. Reuse development checkpoints with hashes recorded in E9_LOCK.json. Stop a part on a runtime failure or default-parity failure. Do not adapt scientific settings.

## Forecast before the first profile

The three archived C lives used about 1,318 CPU seconds each. Budget approximately 4,000–12,000 CPU seconds per new C life under contention and 30–90 minutes per run as in the brief. The full default profile will replace this rough estimate in E9_PROFILE.json. Six independent, single-threaded CPU workers are the maximum concurrency. Existing baseline and backprop records remain distinct from new executions.

The deployed model has 470,801 parameters at width 512 and an 18-byte window (verify actual count in E9_LOCK.json). Store capacity, optimizer moments, cached prompt replay and development checkpoints belong to the cost account. No biological-plasticity or independent-confirmation claim is made by this compute sweep.
