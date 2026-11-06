# Reproduce the three official development runs

Run in a separate copy of the output tree from the same project root. Use the PyTorch environment and the local V216 and V181 archive, and keep the cached V207 prose file available. Start with an empty `official` directory. The launch manifests under the original `official` tree record all worker arguments. Replace the archive path in copied manifests with your own local archive path.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_USE_SHM=0
r=output/minifly-remaining-experiments
python -B "$r/official_encoder.py" prepare "$archive"
python -B "$r/official_mod97.py" prepare "$archive"
python -B "$r/official_attractor.py" prepare "$archive"
```

- **Encoder.** Run `official_encoder.py develop SEED` for each seed, then `official_encoder.py run ARCHIVE SEED ENCODER` for each seed and encoder in the lock. The latter runs both opcode laws and both prior-history controls.
- **Mod 97.** Run `official_mod97.py run ARCHIVE SEED` for each seed. When all seed jobs finish, run `official_mod97.py sealed ARCHIVE`. Use `mod97_early_branches.py run ARCHIVE SEED` to complete the registered shrinkage contrasts for each seed classified as early transfer. It can recreate the common plateau state from birth. Once all those jobs finish, use `mod97_early_branches.py sealed ARCHIVE` and `mod97_diagnostics.py ARCHIVE`. The shuffled-label continuation is conditional on the delayed-rise decision.
- **Attractor.** Run `official_attractor.py run ARCHIVE SEED TOPOLOGY TEACHING` for each seed, topology and teaching condition. Run `attractor_conditional.py evaluate ARCHIVE` after all 18 histories finish. A passing condition opens `attractor_conditional.py run ARCHIVE SEED` for each seed.
- **Report.** Run `official_report.py` after all applicable scoring and continuation jobs finish. Use `official_report.py progress` for the receipt counts during a run.

Use `launch_jobs.py MANIFEST.json CONCURRENCY` for local scheduling. The original run used independent queues for the three protocols. Each worker used single-threaded PyTorch. Do not run two workers for the same condition at once.
