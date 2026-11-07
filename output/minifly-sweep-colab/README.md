# MiniFly Colab mechanism sweep

The [standalone notebook](../MiniFly_Sweep_Colab.ipynb) includes the full eleven-block development sweep, with its source and compact FlyWire v630 inputs embedded. The [design memo](../minifly-whole-repo-audit/MINIFLY_MECHANISM_SWEEP_DESIGN_20261106.md) describes the candidate comparisons and decision rules.

## Run in Colab

Open the notebook in a standard CPU runtime. In the first code cell set `PROFILE = "screen"` and leave `BLOCKS = []`, then select **Run all**. The screen runs 99 jobs across 11 blocks, three arms and three seeds. The default smoke profile runs 33 jobs across one seed. Set `SAVE_TO_DRIVE = True` to persist receipts in your Google Drive across reconnects.

## Run from this source directory

```sh
python -m pip install -r requirements.txt
python sweep.py run --profile screen --workers 2 --out ./receipts
python sweep.py report --profile screen --out ./receipts
```

Use `--blocks` to select individual blocks. The runner verifies the source and input manifest before running and records the runtime and paired panel hash per job. Completed receipts can be reused with the same source, data, runtime, profile and output directory.

## Archived Colab run

The [scorecard](SCORECARD.md) reports every arm and seed, the main findings, and qualification limits.

The [original receipt ZIP](../MiniFly_outputs_receipts.zip) contains 99 completed screen jobs and 33 completed smoke jobs. All 44 block, seed and profile groups have three arms with identical panel hashes within each group. The notebook and receipts share the source and data manifest SHA-256 `fc40a3417bb3b3ed95c6bed3be865aaaf023a5560f1b9d695a29e576ee845299`.

These are development comparisons. The central-complex block uses the prior hand-set navigation gains. The later [fitted compass and steering confirmation](../minifly-connectome-navigation/FIT_OFFICIAL_REPORT.md) and [FlyWire path-integration exercise](../minifly-connectome-navigation/PI_REPORT.md) are separate runs.
