# Reproduce the revised ordered-byte run

[Design and endpoint](CAMPAIGN_DESIGN.md) · [Source lock and qualification](campaign/LOCK.json) · [Live checkpoint summary](campaign/SUMMARY.json)

The following assumes the working directory is the MiniFly project root and Python is from the PyTorch environment. Set `archive` to the local archive containing the locked V216 and V181 dependencies. The campaign files are local and content hashes are in the lock. Re-running a worker resumes its own saved stage and clocks. Run only one process for a given birth and front end.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_USE_SHM=0
experiment=output/minifly-ordered-front-end
# Prepare only for a new copy with no campaign/LOCK.json.
python -B "$experiment/campaign.py" "$archive" prepare
# The six workers can run concurrently when memory and CPU allow.
for birth in ordered-dev-68071 ordered-dev-80413; do
  for arm in stock ordered bag; do
    python -B "$experiment/campaign.py" "$archive" worker "$birth" "$arm"
  done
done
python -B "$experiment/summarize_campaign.py"
python -B "$experiment/adaptive_audit.py"
python -B "$experiment/vocabulary_diagnostic.py" "$archive"
python -B "$experiment/adaptive_audit.py" --archive "$archive"
python -B "$experiment/campaign_tables.py"
```

Each worker directory holds frozen scores and checkpoints for both opposite teaching histories. `STATUS.json` records the latest durable progress. `A_100k.json` holds raw native responses and their exact targets. The three later JSON receipts hold B, A′ and no-edit responses. Both continuation branches are forked from the same B checkpoint. `A_prime.pt` preserves the edited state. The no-edit state is scored and recorded by hash.

The vocabulary and all-assignment files are supplementary evaluations of frozen 100,000-record checkpoints. The adaptive shortcut audit evaluates the benchmark and does not touch model weights.

## GitHub package

The packaged Python source and lock preserve the run layout. Run the commands above from this package's root. Large raw JSON receipts are compressed losslessly. To regenerate the campaign summary from the packaged receipts, decompress those files first.

```sh
find output/minifly-ordered-front-end -name '*.json.gz' -exec gzip -dk {} +
python -B output/minifly-ordered-front-end/summarize_campaign.py
```

The [checkpoint manifest](campaign/CHECKPOINT_HASHES.json) records checksums for the local PyTorch snapshots. A fresh training run in a separate copy generates those files. Use the archived V216 and V181 dependencies identified in the source lock.
