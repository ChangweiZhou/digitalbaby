# E24 host-check wave (11 November 2026): T-ans reproduces exactly, D-ans does not. Stopped.

**Brief:** `OPUS_TASK_E24_EVERY_BYTE_CAUSE_20261111.md`, revision 2. The coordinator released the host-check wave only.

**Run:**
- Sources: the codex branch at f4c6aa8c.
- Machine: a four-core cloud Xeon (2.10 GHz), Python 3.11.18, torch 2.15.0+cpu, one thread per process.
- Work folder: the harness sources, byte-identical to f4c6aa8c (`work_sources_sha256.txt`), with an `exp45` holding only `BIRTH_MANIFEST.json`.
- Both births ran together, from 21:28:37 UTC, with only the brief's flags (`environ.txt`). No other birth ran.

The files are in `exp45/archive_e24_host/cloud_20261111/`.

## Admission

`python3 e24_check.py` exited 0, with `"pass": true` (`e24_check_stdout.txt`, `E24_CHECK_cloud.json`).

- **Bit-for-bit against the harness before the option was added:** equal for all four cells, D and T, answer and every-byte.
- **Against the coordinator's record:** a few fields differ in the last float digits, about 1e-7.

## Each check wrote a new receipt

| Birth | Exit | Receipt written (UTC) | Final `cpu` | Fields the old harness did not write |
|---|---|---|---:|---|
| D-ans 73129 (`E16_ARCH=bank2 E16_BOOKDECAY=off ... dev 73129`) | 0 at 22:22:41 | 22:22:40 | 3,203 s | `complete`, `weights_sha256`, `loss`, `decay` and others |
| T-ans 73129 (the same plus `E16_BIRTHRULE=T E16_TRACEUP=all`) | 0 at 22:23:51 | 22:23:50 | 3,272 s | `complete`, `weights_sha256` and others |

## Comparison with the pinned originals

The scripts are `compare.py` and `compare_*.json`.

| | T-ans 73129 against `e17_dev_bank2_nbd_bT_tall_73129` (54f14b02… / e679f516…) | D-ans 73129 against `e17_dev_bank2_nbd_73129` (e1e0826e… / 8ee5e4e6…) |
|---|---|---|
| Sealed scores, six checkpoints | **identical at all six** | **different at all six** |
| Training loss per checkpoint | identical | differs from the first checkpoint (4,096 updates: 0.5585 against 0.5642) |
| Weights | **24 of 24 tensors bit-equal**; file hash e679f516… equals the original's | 1 of 24 tensors equal (`R`); largest difference 2.58; hash 08dc656a… against 8ee5e4e6… |
| Hard mean, add89 and sub89 at k = 12, by checkpoint | 9.8, 36.5, 59.7, 76.0, 83.7, 87.5, the same as the original | 16.8, 54.5, 84.3, 87.7, 91.8, **93.2**, against the original's 20.3, 52.5, 77.5, 87.2, 90.7, **91.2** |

## Verdict

**D-ans differs from its original, so the wave stops here, as the brief and the coordinator's instruction require.** No D-all, T-all or T-bal birth was started.

**Not established here: why D-ans differs.**
- T-ans's original is E18's B0 birth (commit 2eafb07b). It was made on this cloud host type with the merged harness, and it reproduces bit for bit.
- D-ans's original came from E17 Phase 2 (commit fad2cec7, 9 November). The manifest says it "predates hashed receipts", and its receipt records only `arch`, `seed`, `kmax` and `bookdecay`.
- The rerun differs from the very first checkpoint, so the cause is in the configuration or the code path, not late numerical drift. That points to the harness or host that made the original, not to this host.
- No diagnosis was run.

## Pace and forecast

- **Measured:** 3,203 and 3,272 CPU-s per birth, two processes on four cores, 54–55 min of wall time each.
- **Forecast for the nine new births, if released:** about 3,300–3,600 CPU-s each with three at a time, about 30,000–32,000 CPU-s in all. That is three waves of about an hour, about 3 h of wall time.
  - That compares with the brief's 3,500–4,500 each and about 45,000 for eleven.
  - The every-byte loss changes only which positions carry the loss, so no large cost change is expected; this has not been measured.

## Not committed

The harness's intermediate `.ckpt` and `.ckpt.prev` files (13.5 MB each) are not committed. They are kept in the session's scratchpad, and their `.ckpt.log` files are committed.
