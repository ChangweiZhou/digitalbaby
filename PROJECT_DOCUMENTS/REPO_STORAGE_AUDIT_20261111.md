# Repository storage audit (Job S3), 11 November 2026

Job S3 of `OPUS_TASK_SIDE_JOBS_READ_ONLY_20261110.md`. Read-only: nothing in the repository was deleted, moved, rewritten, repacked or pushed. Measured 11 November 2026, 00:15 to 00:45 UTC, by a Claude Opus 5.5 worker.

**Measurement point.** Default branch `codex/minifly-consolidated-20261023` at `df09c03ad2` (10 Nov 19:07 EST). During the audit the remote moved three commits on, to `40cb1c208a` (E22 brief and dry run). Those three commits change 8 text files and add no weights; the clone tests in section 4 ran against `40cb1c208a`. Every other number is at `df09c03ad2`.

**Units.** MiB = 2^20 bytes, GiB = 2^30 bytes. "Uncompressed" is the blob size, which equals the file size in a checkout. "Packed" is the size on disk inside git's pack files, after zlib compression and deltas.

## Summary of the numbers

| Quantity | Value |
|---|---|
| Object store on GitHub (`size` field of the repository API) | 10,530,576 KiB, which is 10.04 GiB |
| Object store in this session's clone (`git count-objects`) | 10.04 GiB packed, 49,632 objects, 3 packs |
| Commits reachable from every remote ref (12 branches, 3 PR refs, 37 tags) | 1,171; 893 of them on the default branch |
| Unique blobs in all history | 37,776; 11.26 GiB uncompressed (12,088,396,772 B); about 10.04 GiB packed |
| Checkout of the default branch | 35,345 files, 11.12 GiB (11,944,776,482 B); `du` on this checkout gives 12 G |
| Full clone plus checkout, as each container gets now | about 21 GiB (10.04 GiB packed + 11.12 GiB checkout) |
| Blobs that are not at the tip of any branch or PR | 1,835 blobs, 33.2 MB uncompressed, **4.7 MB packed (0.04 % of the store)** |
| Blobs that are not at the tip of the default branch | 8,159 blobs, 268.8 MB uncompressed, 68.4 MB packed (0.6 %) |
| Share of the store that is at the tip of the default branch | 29,617 unique blobs, 9.98 GiB packed, which is **99.4 %** |
| Zip bundles (`.zip` and `.zip.partNN`) at the tip of the default branch | 185 files, 4.31 GiB, 38.7 % of the checkout |
| Of that, content that also sits unzipped next to the bundle (same name, size and CRC-32) | 1.67 GiB of compressed zip content |
| Model weights `.pt` at the tip of the default branch | 239 files, 895.5 MiB (7.9 %); 237 of them in `MOD97_LOCAL_RULE_20261106/exp45/` |
| Bytes added since 1 November | 1.26 GiB uncompressed, 0.89 GiB packed; `.pt` is 878.6 MiB of it |
| Commits since 1 November that are partial or in-progress receipts | 221 of 565 (39 %); they added 147.8 MiB |
| Shallow clone of the default branch at depth 1 | not run: it would need about 21 GiB and the session had 8.8 GB free (section 4) |
| **Tested:** shallow sparse clone of `MOD97_LOCAL_RULE_20261106/` without `.pt`, plus root files | **75 MiB `.git` + 200 MiB checkout = 275 MiB**; 3,978 files |
| **Tested:** the same plus `.claude/` and `PROJECT_DOCUMENTS/` | 85 MiB `.git` + 255 MiB checkout = 340 MiB; 5,472 files |
| **Tested:** whole tree without `.pt`, `.zip`, `.zip.part*`, `.npz` and `.gz` | 135 MiB `.git` + 839 MiB checkout = 975 MiB; 25,097 files |

**What the numbers mean.** The repository is large because of what sits at the tip of the default branch, not because of old history. Shortening history does not help: this session's clone is shallow (depth 50) and still holds 10.04 GiB, all but 2.9 MB of the blobs in the repository. A history rewrite that removed only blobs no branch still holds would save 4.7 MB. Most of the bytes come from the September and October publication bundles, which are zip archives split into parts below GitHub's 100 MB file limit, and from `.json.gz` and `.npz` receipts. Since 1 November, growth comes mostly from `.pt` weights in `MOD97_LOCAL_RULE_20261106/exp45/`.

## How the numbers were measured

- **History.** A blob-less mirror (`git clone --filter=blob:none --mirror`) was cloned into the scratchpad. It took 2.6 MiB and holds every commit and tree of all refs. It gives each blob's paths and the first commit that added it (`git log --all --raw --no-renames -m --root`), with no blob downloaded. In that clone, `git rev-list --objects --all --missing=print` lists every blob in history: 37,776.
- **Blob sizes.** Sizes come from `git cat-file --batch-all-objects --batch-check` on this session's own clone, which is shallow but holds 37,697 of the 37,776 blobs. The other 79 blobs are 57 `.md`, 20 `.py` and 2 `.json`, in `MOD97_LOCAL_RULE_20261106/`, `PROGRAM_STATE_20261107.md`, the past-work review and `output/`. They are reachable only from commits beyond the shallow boundary. Their sizes come from GitHub's tree API (`gh api repos/.../git/trees/<sha>`), which gives a size for each blob without downloading it. Together they are 2,890,621 B. **So every blob in history has an exact uncompressed size.** Packed sizes are this clone's `%(objectsize:disk)`. A fresh pack on GitHub, or in another clone, could delta differently. The total agrees with GitHub's `size` field to within 0.1 %.
- **Tip sets.** These come from `git ls-tree -r` on every branch, PR ref and tag in the blob-less mirror. Tags add almost nothing beyond the branches: 6 blobs.
- **Dates.** "Added on day X" means the committer date (in the committer's time zone) of the earliest commit, in `--date-order`, whose diff adds or changes the blob. Rebased copies of a commit keep the earliest date. Repeated subjects account for 35 extra commits. Most look like rebased copies, though that was inferred from the subjects and not checked. Each copy counts as a commit in the commit counts.
- **No lazy fetch happened.** This session's clone is not a promisor clone, and its pack stayed at 10,533,916 KiB throughout.

## 1. Object store and checkout, by top-level folder and by file type

### By top-level folder, at the tip of the default branch (252 top-level entries)

| Folder | Files | MiB in checkout | MiB packed, all history |
|---|---|---|---|
| `next_core_budget_v2_20261006/` | 2,117 | 1,548.6 | 1,422.6 |
| `MOD97_LOCAL_RULE_20261106/` | 4,169 | 1,083.6 | 845.7 |
| `minifly_stop_only_confirm_v1_20261023/` | 2,929 | 941.9 | 806.3 |
| `arithmetic_value_interaction_dev_v1_20261012/` | 976 | 735.9 | 725.4 |
| `persistent_core_v2_trial/` | 1,649 | 715.8 | 711.4 |
| `minifly_address_capacity_v1_dev_20261023/` | 812 | 670.2 | 580.4 |
| `native_context_binding_dev_v2_1_20261017/` | 576 | 508.0 | 422.7 |
| `minifly_a_v3/` | 1,029 | 467.0 | 453.9 |
| `minifly_v3_dev_20261020/` | 368 | 422.3 | 396.5 |
| `addition_feedback_efficiency_dev_v1_20261011/` | 1,644 | 368.3 | 365.2 |
| `minifly_b/` | 979 | 323.2 | 319.4 |
| `minifly_recoverability_v1_dev_20261022/` | 783 | 321.4 | 264.4 |
| `addition_uncertainty_restoration_dev_v1_20261011/` | 509 | 251.4 | 210.6 |
| `arithmetic_mechanism_transfer_dev_v2_20261012/` | 1,474 | 236.5 | 232.4 |
| `minifly_recoverability_v1_r2_dev_20261023/` | 490 | 203.8 | 141.5 |
| 10 more folders, each 140 to 203 MiB | | 1,733.3 | |
| the other 227 entries | | 864.3 | |
| `PROJECT_DOCUMENTS/` (for reference) | 1,469 | 55.1 | |
| `.claude/` (for reference) | 24 | 0.3 | |

In the last column, history bytes are attributed to the folder of each blob's first path.

Within `MOD97_LOCAL_RULE_20261106/`, `exp45/` alone is 1,051.5 MiB: 884.7 MiB of `.pt` and 125.7 MiB of `.json`. The rest of the folder is 32 MiB.

### By file type

| Type | Files at the default tip | MiB in checkout | Share | Unique blobs in history | MiB uncompressed, history | MiB packed, history |
|---|---|---|---|---|---|---|
| `.zip` and `.zip.partNN` (result and preparation bundles) | 185 | 4,408.8 | 38.7 % | 190 | 4,414.6 | 4,362.4 |
| `.gz` (2,522 MiB of it `.json.gz`, 167 `.jsonl.gz`, 69 `.pkl.gz`) | 4,775 | 2,758.7 | 24.2 % | 4,942 | 2,759.9 | 2,734.0 |
| `.npz` (arrays, including older `*_W.npz` weights) | 5,053 | 2,489.1 | 21.9 % | 4,494 | 2,513.8 | 2,252.8 |
| **`.pt`** (PyTorch weights) | 239 | 895.5 | 7.9 % | 237 | 889.4 | 778.6 |
| **`.json`** | 13,017 | 694.7 | 6.1 % | 16,860 | 712.2 | 57.6 |
| **`.log`** | 5,000 | 7.4 | 0.1 % | 3,946 | 7.9 | 1.0 |
| `.py` | 3,004 | 22.3 | 0.2 % | 2,152 | 22.1 | 6.6 |
| `.md` | 1,664 | 17.8 | 0.2 % | 2,117 | 36.2 | 8.5 |
| `.jsonl` | 45 | 6.5 | 0.1 % | 1,086 | 63.5 | 5.1 |
| other | 2,363 | 90.6 | 0.8 % | 1,752 | 108.8 | 77.8 |
| **Total** | 35,345 | 11,391.4 | | 37,776 | 11,528.4 | 10,281.6 |

In the history columns, each blob is counted once and typed by its first path. The file counts in the two halves differ because one blob can sit at several paths. Plain `.json` compresses about 12 to 1 in the pack; the zips, `.gz`, `.npz` and `.pt` barely compress at all.

`.gitignore` lists `*.pt` and `*.pth`, so every `.pt` in the repository was added with `git add -f` or an equivalent. There is no `.gitattributes` and no Git LFS. One GitHub release exists, `v0.66`, with a 756.9 MB asset `genesis_v66_checkpoint.pt`. That is a precedent for keeping weights as release assets.

## 2. The fifty largest blobs in history, and history no branch still holds

All fifty are zip bundles or zip parts from 29 September to 20 October. All fifty are still at the tip of the default branch, each at exactly one path. Together they are 2.67 GiB, which is 23.7 % of history. The largest `.pt` is 6.8 MiB (a dry run in `exp45/archive_e18_dry/`), the largest `.npz` 7.0 MiB and the largest `.json` 44.6 MiB (`minifly_address_capacity_v1_dev_20261023/SAVED_CONTROL_ROWS.json`).

| # | MiB | blob | added in commit | date | path (first) | at default head |
|---|---|---|---|---|---|---|
| 1 | 90.0 | `a9dc977b58` | `d4e45782bc` | 2026-09-29 | `minifly_b/return_bundle/minifly_b_return_bundle.zip.part00` | yes |
| 2 | 75.0 | `299a4d8b24` | `a555ae0616` | 2026-10-11 | `addition_feedback_efficiency_dev_v1_20261011/results/publication/RESULT_BUNDLE.zip.part00` | yes |
| 3 | 75.0 | `1bf2d2454d` | `a555ae0616` | 2026-10-11 | `addition_feedback_efficiency_dev_v1_20261011/results/publication/RESULT_BUNDLE.zip.part03` | yes |
| 4 | 75.0 | `eda359031f` | `a555ae0616` | 2026-10-11 | `addition_feedback_efficiency_dev_v1_20261011/results/publication/RESULT_BUNDLE.zip.part02` | yes |
| 5 | 75.0 | `8fb52404be` | `a555ae0616` | 2026-10-11 | `addition_feedback_efficiency_dev_v1_20261011/results/publication/RESULT_BUNDLE.zip.part01` | yes |
| 6 | 72.7 | `edfeb05cad` | `66f304caf9` | 2026-10-12 | `arithmetic_representation_learning_dev_v1_20261012/results/RESULT_BUNDLE.zip` | yes |
| 7 | 71.1 | `21f5e0f5cd` | `de39acef92` | 2026-10-11 | `addition_value_retention_confirm_v1_20261011/results/RESULT_BUNDLE.zip` | yes |
| 8 | 70.6 | `69b3ddf758` | `d4e45782bc` | 2026-09-29 | `minifly_b/return_bundle/minifly_b_return_bundle.zip.part01` | yes |
| 9 | 66.4 | `f2c0cfc8a2` | `f49a74f2a5` | 2026-10-11 | `addition_uncertainty_restoration_dev_v1_20261011/results/RESULT_BUNDLE.zip` | yes |
| 10 | 64.0 | `0556bb4b97` | `b9415ae9c5` | 2026-10-13 | `arithmetic_value_interaction_dev_v1_20261012/results/RESULT_BUNDLE_parts/RESULT_BUNDLE.zip.part004` | yes |
| 11 | 64.0 | `a9c75ea75f` | `b9415ae9c5` | 2026-10-13 | `arithmetic_mechanism_transfer_dev_v2_20261012/results/RESULT_BUNDLE_parts/RESULT_BUNDLE.zip.part001` | yes |
| 12 | 64.0 | `08d7c507fd` | `b9415ae9c5` | 2026-10-13 | `arithmetic_value_interaction_dev_v1_20261012/results/RESULT_BUNDLE_parts/RESULT_BUNDLE.zip.part001` | yes |
| 13 | 64.0 | `de61671a05` | `b9415ae9c5` | 2026-10-13 | `arithmetic_value_interaction_dev_v1_20261012/results/RESULT_BUNDLE_parts/RESULT_BUNDLE.zip.part005` | yes |
| 14 | 64.0 | `2a6f10aa44` | `b9415ae9c5` | 2026-10-13 | `arithmetic_value_interaction_dev_v1_20261012/results/RESULT_BUNDLE_parts/RESULT_BUNDLE.zip.part003` | yes |
| 15 | 64.0 | `43308613d6` | `b9415ae9c5` | 2026-10-13 | `arithmetic_value_interaction_dev_v1_20261012/results/RESULT_BUNDLE_parts/RESULT_BUNDLE.zip.part002` | yes |
| 16 | 60.7 | `879b1ebf0b` | `de39acef92` | 2026-10-11 | `addition_efficiency_dev_v1_20261011/RESULT_BUNDLE.zip` | yes |
| 17 | 48.0 | `2f2f25f632` | `18874ca8d8` | 2026-10-17 | `native_context_binding_dev_v2_1_20261017/results/bundle_parts/RESULT_BUNDLE.zip.part02` | yes |
| 18 | 48.0 | `541e45c13e` | `946fc6c8a4` | 2026-10-11 | `addition_sample_generalization_dev_v2_20261011/results/bundle_parts/RESULT_BUNDLE.zip.part03` | yes |
| 19 | 48.0 | `cb98c9c572` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part008` | yes |
| 20 | 48.0 | `816666c615` | `2b9aa7570c` | 2026-10-07 | `associative_retrieval_v1_20261007/publication/archive/RESULT_BUNDLE.zip.part002` | yes |
| 21 | 48.0 | `980112f1c1` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part011` | yes |
| 22 | 48.0 | `176c6b7953` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part001` | yes |
| 23 | 48.0 | `4c6885dce9` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part009` | yes |
| 24 | 48.0 | `371203a4d6` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part007` | yes |
| 25 | 48.0 | `0028a69f2e` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part014` | yes |
| 26 | 48.0 | `2c7c092988` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part012` | yes |
| 27 | 48.0 | `b99a7daaf8` | `c7fa55d562` | 2026-10-17 | `native_context_binding_dev_v2_1_20261017/results/bundle_parts/RESULT_BUNDLE.zip.part04` | yes |
| 28 | 48.0 | `4c416d223c` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part005` | yes |
| 29 | 48.0 | `3c9c2d3d3c` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part006` | yes |
| 30 | 48.0 | `c67d9b1029` | `878fa14a42` | 2026-10-17 | `native_context_binding_dev_v2_1_20261017/results/bundle_parts/RESULT_BUNDLE.zip.part05` | yes |
| 31 | 48.0 | `965d644a6f` | `946fc6c8a4` | 2026-10-11 | `addition_sample_generalization_dev_v2_20261011/results/bundle_parts/RESULT_BUNDLE.zip.part01` | yes |
| 32 | 48.0 | `c7991f665d` | `2b9aa7570c` | 2026-10-07 | `associative_retrieval_v1_20261007/publication/archive/RESULT_BUNDLE.zip.part001` | yes |
| 33 | 48.0 | `16a911a600` | `dfa6926cfb` | 2026-10-17 | `native_context_binding_dev_v2_1_20261017/results/bundle_parts/RESULT_BUNDLE.zip.part07` | yes |
| 34 | 48.0 | `919b180479` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part003` | yes |
| 35 | 48.0 | `8ea3b4f361` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part002` | yes |
| 36 | 48.0 | `89f4756d13` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part004` | yes |
| 37 | 48.0 | `3f919f4ad7` | `51af9ca7b8` | 2026-10-17 | `native_context_binding_dev_v2_1_20261017/results/bundle_parts/RESULT_BUNDLE.zip.part06` | yes |
| 38 | 48.0 | `447d54bbf8` | `3578b3c0f7` | 2026-10-17 | `native_context_binding_dev_v2_1_20261017/results/bundle_parts/RESULT_BUNDLE.zip.part03` | yes |
| 39 | 48.0 | `3a6a70d548` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part013` | yes |
| 40 | 48.0 | `424df89d14` | `595445bc4a` | 2026-10-17 | `native_context_binding_dev_v2_1_20261017/results/bundle_parts/RESULT_BUNDLE.zip.part08` | yes |
| 41 | 48.0 | `00a0da7b8c` | `946fc6c8a4` | 2026-10-11 | `addition_sample_generalization_dev_v2_20261011/results/bundle_parts/RESULT_BUNDLE.zip.part02` | yes |
| 42 | 48.0 | `ab3361e917` | `564bbd56cb` | 2026-10-06 | `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip.part010` | yes |
| 43 | 48.0 | `0722644eac` | `2b9aa7570c` | 2026-10-07 | `associative_retrieval_v1_20261007/publication/archive/RESULT_BUNDLE.zip.part000` | yes |
| 44 | 48.0 | `cc9e110b29` | `924184d7e1` | 2026-10-17 | `native_context_binding_dev_v2_1_20261017/results/bundle_parts/RESULT_BUNDLE.zip.part01` | yes |
| 45 | 46.0 | `eb62b6172d` | `b9415ae9c5` | 2026-10-13 | `arithmetic_mechanism_transfer_dev_v2_20261012/results/RESULT_BUNDLE_parts/RESULT_BUNDLE.zip.part002` | yes |
| 46 | 45.0 | `3138a487f0` | `adb210772e` | 2026-10-14 | `arithmetic_feedback_rule_discovery_dev_v1_20261013/PREPARATION_BUNDLE_parts_20261014/PREPARATION_BUNDLE.zip.part002` | yes |
| 47 | 45.0 | `6454add79a` | `07c21218b3` | 2026-10-20 | `minifly_v3_dev_20261020/publication/RESULT_BUNDLE.zip.part03` | yes |
| 48 | 45.0 | `82ae61ac5e` | `adb210772e` | 2026-10-14 | `arithmetic_feedback_rule_discovery_dev_v1_20261013/PREPARATION_BUNDLE_parts_20261014/PREPARATION_BUNDLE.zip.part004` | yes |
| 49 | 45.0 | `6791259884` | `adb210772e` | 2026-10-14 | `arithmetic_stream_transfer_dev_v1_20261013/PREPARATION_BUNDLE_parts_20261014/PREPARATION_BUNDLE.zip.part003` | yes |
| 50 | 45.0 | `26c5d1dfc9` | `adb210772e` | 2026-10-14 | `arithmetic_model_feedback_dev_v1_20261014/PREPARATION_BUNDLE_parts_20261014/PREPARATION_BUNDLE.zip.part001` | yes |

**Commits that brought the most bytes:** 10 commits brought 59.0 % of all blob bytes, 20 brought 74.5 %, 50 brought 87.7 % and 100 brought 93.2 %. The largest are `3065bb1ae1` (23 Oct, 1,141 MiB, RECOVERABILITY V1-R2 and STOP-ONLY), `b9415ae9c5` (13 Oct, 906 MiB), `adb210772e` (14 Oct, 743 MiB), `5a91144dca` (14 Oct, 724 MiB, "Preserve remaining experiment evidence (2/3)"), `7b64ae252f` (3 Oct, 715 MiB), `564bbd56cb` (6 Oct, 698 MiB) and `1655015158` (23 Oct, 665 MiB).

**History that no branch still holds:** 1,835 blobs, 33.2 MB uncompressed and 4.7 MB packed, are in no branch or PR tip. 1,829 of them are in no tag either. By type they are 18.6 MiB of `.md`, 9.0 MiB of `.json` and 3.5 MiB of `.py`: superseded versions of text files. 8,159 blobs, 68.4 MB packed, are missing from the tip of the default branch but may sit on other branches. **No large file was ever deleted. Everything ever added is still checked out.**

## 3. Growth by day since 1 November, and partial receipts

"Partial receipt" means a commit whose subject matches `partial receipt|in-progress receipt|receipts in progress|in progress` (case-insensitive). Examples are "E19 / E16-M4 partial receipts: E16-M4 checkpoint 3" and "E20-A stage 1: in-progress receipts of the three window-reference lives". The pattern is my choice, not a project definition. It catches some commits about work in progress that carry no receipt, and it misses receipts that are partial but not labelled as such.

| Day (committer time zone) | Commits | Partial-receipt commits | New blob MiB, uncompressed | New MiB packed | New MiB in partial-receipt commits |
|---|---|---|---|---|---|
| 2026-11-01 | 5 | 0 | 0.8 | 0.2 | 0.0 |
| 2026-11-02 | 18 | 0 | 4.9 | 1.1 | 0.0 |
| 2026-11-03 | 42 | 0 | 8.6 | 2.5 | 0.0 |
| 2026-11-04 | 21 | 0 | 36.2 | 14.2 | 0.0 |
| 2026-11-05 | 13 | 0 | 149.5 | 34.1 | 0.0 |
| 2026-11-06 | 28 | 0 | 25.5 | 8.8 | 0.0 |
| 2026-11-07 | 20 | 2 | 52.5 | 41.0 | 0.1 |
| 2026-11-08 | 78 | 8 | 186.5 | 97.6 | 0.8 |
| 2026-11-09 | 165 | 109 | 331.2 | 279.5 | 59.6 |
| 2026-11-10 | 175 | 102 | 492.6 | 428.0 | 87.3 |
| **1 to 10 November** | **565** | **221 (39 %)** | **1,288.3 (1.26 GiB)** | **911 (0.89 GiB)** | **147.8 (11.5 %)** |
| Before 1 November | 606 | 10 | 10.00 GiB | 9.16 GiB | |

What the 1 to 10 November growth is made of:

- **By type:** `.pt` 878.6 MiB, `.json` 166.0, `.npz` 66.5, `.gz` 58.6, `.jsonl` 57.1, `.md` 22.2, `.zip` 9.8 and `.py` 7.2.
- **By folder:** `MOD97_LOCAL_RULE_20261106/exp45` 1,026.3 MiB, `minifly_v217_statelaw_reproduction` 66.4, `minifly_v222_reentry_continuation` 57.6 and `minifly_v207_prose_pretraining` 24.8; the rest is small.

The store grew about 0.9 GiB packed in ten days, after 9.2 GiB before 1 November. The rate has been rising: 0.28 and 0.43 GiB packed on 9 and 10 November. Commits on those two days are 62 % partial receipts by count (211 of 340), but partial receipts carry only 18 % of those days' bytes. Most of the bytes come with the completion commits for weights and logs.

## 4. What a new worker needs: tested clone commands

All tests ran in the session scratchpad against `https://github.com/ChangweiZhou/digitalbaby.git`, default branch at `40cb1c208a`. A watchdog would have aborted any clone that passed 2 GB; none came close. Every test clone was deleted afterwards.

**(a) Shallow clone of the whole default branch at depth 1: not run, because it does not fit.** The command would be:

```
git clone --depth 1 --single-branch --branch codex/minifly-consolidated-20261023 https://github.com/ChangweiZhou/digitalbaby.git
```

It would download every blob at the tip, which is 9.98 GiB packed in this clone's pack, and write an 11.12 GiB checkout: about 21 GiB in all. The session had 8.8 GB free. **Depth saves almost nothing here, because 99.4 % of the store is at the tip.** That is why this session's depth-50 clone still came to 10.04 GiB. The 9.98 GiB figure is measured on this clone's pack. A fresh depth-1 pack from GitHub could differ by the deltas, but not by much, since the large files barely compress or delta.

**(b) Shallow sparse clone of `MOD97_LOCAL_RULE_20261106/` without weights, plus the files at the root. Tested.**

```
git clone --depth 1 --filter=blob:none --sparse --single-branch \
    --branch codex/minifly-consolidated-20261023 \
    https://github.com/ChangweiZhou/digitalbaby.git work
cd work
git sparse-checkout set --no-cone '/*' '!/*/' '/MOD97_LOCAL_RULE_20261106/' '!*.pt'
```

| Step | Time | `.git` | Checkout | Files |
|---|---|---|---|---|
| Clone, which checks out only the files at the root | 1.8 s | 6.8 MB | 0.8 MB | 43 |
| After `sparse-checkout set` | 4.7 s | 78.6 MB (75 MiB) | 209.4 MB (200 MiB) | 3,978, no `.pt` |

The total is about 288 MB. The `.git` here holds only the blobs that were checked out, plus the commit and trees. The checkout matches the prediction from the blob table: `MOD97_LOCAL_RULE_20261106/` without `.pt` is 3,932 files and 198.9 MiB.

**(c) The same plus the skills and the project documents (`AGENTS.md` sends workers to `.claude/skills/...` and `PROJECT_DOCUMENTS/...`). Tested by widening (b).**

```
git sparse-checkout set --no-cone '/*' '!/*/' '/MOD97_LOCAL_RULE_20261106/' '/.claude/' '/PROJECT_DOCUMENTS/' '!*.pt'
```

This took 2.3 s. The result is 89.5 MB `.git` and 267.4 MB checkout: about 357 MB and 5,472 files.

**(d) The whole tree without binary formats. Tested in a fresh clone.** This suits a worker who must read receipts in every folder.

```
git clone --depth 1 --filter=blob:none --sparse --single-branch \
    --branch codex/minifly-consolidated-20261023 \
    https://github.com/ChangweiZhou/digitalbaby.git work
cd work
git sparse-checkout set --no-cone '/*' '!*.pt' '!*.zip' '!*.zip.part*' '!*.npz' '!*.gz'
```

This took 17.2 s. The result is 142.1 MB `.git` and 880.1 MB checkout: about 1.0 GB and 25,097 files. The prediction was 839.3 MiB, 25,093 files.

Notes on these clones:

- **Fetching a weight later.** A worker in clone (b), (c) or (d) that needs a weight can fetch it on demand: `git sparse-checkout add 'MOD97_LOCAL_RULE_20261106/exp45/<file>.pt'` with a `!`-free pattern for that path, or `git show HEAD:<path> > <file>`. Either fetches only that blob, because the clone is a promisor clone. This was not tested beyond the commands above.
- **Committing and pushing.** A worker can commit and push from such a clone; git only needs the tip it builds on. This was not tested, because pushing was out of scope.
- **The container's own first clone.** It is made by the cloud platform before the session starts: here depth 50 with every blob. Whether the environment can make that first clone sparse or blob-less was not tested and is outside this audit. A worker can still make the small clone itself, beside or instead of the one it was given, only if the platform's clone is not already filling the disk.

## 5. Duplicates and weights no later job has read

### Identical blobs under several paths

- **At the tip of the default branch:** 1,784 blobs sit at more than one path, 7,512 paths in all. The extra copies cost 119.4 MiB in the checkout and nothing in the object store, because git keeps one copy of each blob.
- **The largest cases at the tip:**
  - `exp45/e16wr_technical/{default,original}/trajectory.json`: 16.0 MiB, 2 copies.
  - `exp45/e16w2_technical/*_trajectory.json`: 5.1 MiB, 3 copies.
  - `calibration.npz` in three minifly folders: 1.4 MiB, 4 copies.
  - Two dry-run `.pt` weights shared between `archive_e16f6_dry/` and `archive_e16f7_dry/`: 3.1 MiB each.
  - Many `*_W.npz` under `next_core_budget_v2_20261006/.../concurrency4` and `.../concurrency8`.
- **In all history:** 2,764 blobs sit at more than one path. The extra paths add 171.7 MiB to checkouts and nothing to the store. Two `.pt` blobs are among them.
- **Zip bundles that repeat their own unzipped contents.** This is the duplication that does cost store bytes, since git cannot see inside a zip. All 128 bundles at the tip were opened, rejoining the split parts one bundle at a time in the scratchpad and deleting each afterwards. Of the 4,408.8 MiB stored in bundles, **1,714.6 MiB is compressed content of entries that also sit at the tip as plain files** with the same file name, the same size and the same CRC-32. Examples:
  - `arithmetic_value_interaction_dev_v1_20261012/results/RESULT_BUNDLE.zip`: 330 of 330 MiB duplicated.
  - `minifly_v3_dev_20261020/publication/RESULT_BUNDLE.zip`: 201 of 201 MiB.
  - `minifly_b/return_bundle/minifly_b_return_bundle.zip`: 160 of 161 MiB.
  - Others repeat almost nothing. `next_core_budget_v2_20261006/publication/archive/RESULT_BUNDLE.zip`, for one, is 691 MiB with 2 MiB matched.

  Matching uses name, size and CRC-32, not a byte-for-byte comparison, so the 1,714.6 MiB is an upper estimate for exact duplicates. A collision that matches name, size and CRC-32 at once is unlikely.

### Weights that no later job has read

**How I decided.** No record exists of which weight files were opened by later jobs; the logs do not name files they load. So I answered a narrower, checkable question: **does any code in the repository name the file in a load call?**

1. I collected every string literal ending in `.pt` inside `torch.load(...)` or `T.load(...)` calls in all 3,004 `.py` and the `.sh` files at the tip. That gave 27 usable patterns.
2. I turned each pattern into a regular expression. A `{seed}`-like placeholder matches only digits; any other `{...}` placeholder matches one token of letters and digits, with no underscore.
3. I matched these against the 239 `.pt` file names at the tip.

Results for the 239 weights (895.5 MiB):

| Class | Files | MiB | Contents |
|---|---|---|---|
| Matches a load pattern in code (for example `e11_dev_M_{seed}.pt`, `e8_dev_{seed}.pt`, `e17_dev_bank2_nbd_{seed}.pt`, `e19_dev_kbank2_bptt_{seed}.pt`) | 49 | 167.0 | development births that later lives load: `e11_dev` 18, `e12_A_e8_state` 9, `e8_dev` 9, `e17_dev` 6, `e19_dev` 3, `e16_life` 3, `e7_dev` 1 |
| Named as a `.pt` string in code, but not in a load call | 2 | 10.8 | `native_context_binding_dev_v2_1_20261017/.../W_OLD_END.pt`, `W_OLD_REST.pt` |
| **No code names them for loading** | **188** | **717.7** | 51 dry-run weights in `exp45/archive_*dry*/` (202.8 MiB); end-of-life weights `e17_life_*` 51 (219.4 MiB) and `e16_life_*` 63 (192.6 MiB); `e17_dev` 12 (66.7 MiB); `e19_dev` 6 (27.1 MiB); `e7_dev` 5 (9.0 MiB) |

Limits of the method:

- "No code names it" does not prove "never read".
  - A path built some other way would be missed: from a variable, a JSON field (for example `torch.load(O/j['checkpoint'])` in `e14s_campaign.py`), a command-line argument or a glob.
  - So would a load by a tool outside the repository, such as a Mac run, a notebook or Codex's checks.
- A match does not prove a read either: it shows only that some code can load a file of that name.
- The dry-run weights are the clearest case. They are archived evidence that a dry run happened, and the briefs score lives from their JSON receipts, not from dry weights.
- The end-of-life weights (`e16_life_*`, `e17_life_*`) are what a later restart-safe resume or a re-scoring would need. Nothing at the tip loads them now, but that is not evidence they are unneeded.

## 6. Options (set out without choosing)

Three numbers decide what the options can do:

- 99.4 % of the store is at the tip of the default branch.
- 1 to 10 November added 0.89 GiB packed, 879 MiB of it `.pt`.
- Each new container now receives about 21 GiB: a 10 GiB store and an 11 GiB checkout.

### Option 1. Leave the repository as it is; workers use shallow sparse clones

- **Saves:** per worker, about 21 GiB down to 0.29 GB (clone b), 0.36 GB (c) or 1.0 GB (d), as measured above. It does nothing for the repository on GitHub. Growth continues at the current rate, about 0.4 GiB per day packed on the busiest days.
- **Breaks:**
  - Nothing in the repository or its history.
  - A worker in a sparse clone does not see excluded files. It must fetch a weight on demand before loading it, and a script that globs `*.pt` finds nothing. Folder-wide checks may report missing files that are only unchecked out.
  - The cloud container's own first clone stays full unless the platform can be told otherwise (untested). A worker that makes a sparse clone beside a full one saves nothing on disk.
- **Who acts:**
  - The coordinating session puts the commands into briefs.
  - Each worker runs them.
  - Mr. Zhou or whoever controls the environment decides whether the environment's first clone can change. That is outside this audit.
- **Reversible:** yes, fully. A sparse clone can be widened at any time (`git sparse-checkout disable`, or `git fetch --unshallow` where the disk allows).

### Option 2. Stop committing weights except at completion, and never for dry runs

- **Saves:** future growth only; nothing already stored. The upper bound comes from 1 to 10 November:
  - About 879 MiB of 1,288 MiB came as `.pt`. That counts weights added at completion as well, so the realistic saving is smaller.
  - Dry-run weights alone are 51 files, 202.8 MiB at the tip.
  - Partial-receipt commits added 147.8 MiB, `.pt` and text together.
- **Breaks:**
  - Mid-run weights committed as restart insurance would be lost when a container restarts. The 10 November E16-M4 restart is the case in point; the E21 brief, "checkpoint at every scored stage", may rely on such checkpoints.
  - Any check that reproduces a dry run from its archived weight, such as "an archived dry birth reproduces after the change" in the E18b commit, would need the weight kept elsewhere or regenerated.
  - It changes a process rule, so under the governance skill it needs Mr. Zhou's explicit approval and a separate, reviewed commit.
- **Who acts:**
  - Mr. Zhou approves.
  - The coordinating session writes the rule into the waiting briefs.
  - Codex and Claude workers follow it.
  - A pre-commit check (for example, refusing `.pt` paths that contain `dry`) is optional and would itself be a protected change.
- **Reversible:** yes. It is a rule about future commits, and nothing is removed.

### Option 3. From now on, move weights to release assets or Git LFS

- **Saves:** future weight bytes in the git store; with Option 2's rule, nearly all of the current growth. Nothing already stored is saved, unless history is also rewritten (Option 4).
- **Release assets:**
  - A release asset may be up to 2 GiB per file, under GitHub's published limits (not re-checked today).
  - There is a precedent: release `v0.66` holds a 757 MB `.pt`.
  - The repository would keep a manifest per run: file name, sha256, release tag. Any loader would need a download step.
- **Git LFS:**
  - Git would store small pointer files and LFS would keep the content.
  - LFS storage and bandwidth count against an account quota. The quota and its cost for this account were not measured.
- **Breaks:**
  - Code that loads `OUT / f"e8_dev_{seed}.pt"` from the working tree would find nothing, with release assets, or a pointer text file, with LFS when `git-lfs` is not installed, until a fetch step runs.
  - Mr. Zhou's other tools that read weights by path from a clone would need the same step. Which tools those are is not known to this audit.
  - Hashes cited in receipts stay valid: the commit and the weight's sha256 do not change, but the weight no longer comes with the commit.
  - LFS also changes what a plain `git clone` contains for anyone without `git-lfs`.
- **Who acts:**
  - Mr. Zhou: repository settings, LFS billing, and creating releases (or granting a token that can).
  - The coordinating session: manifest format and briefs.
  - Workers: uploading at completion.
- **Reversible:** going forward, yes. Files can be committed directly again later. Turning on LFS for past files (`git lfs migrate import`) is a history rewrite and belongs to Option 4.

### Option 4. Rewrite history (described, not recommended)

- **What it would take to save anything:**
  - Removing only what no branch holds saves 4.7 MB, about nothing.
  - A real saving needs two steps. First, delete files from the tip in ordinary commits: the zip bundles (4.31 GiB, of which 1.67 GiB repeats files already unzipped beside them), the `.json.gz` and `.npz` receipts, and the weights. Then purge them from all history with a tool such as `git filter-repo`.
  - Purging only the bundles from history would shrink the store from 10.04 GiB to about 5.8 GiB packed (bundles are 4,362.4 MiB packed). Purging every `.pt` as well would remove a further 0.76 GiB.
  - Neither step was tested.
- **Breaks:**
  - **Every commit hash from the first rewritten commit onward changes.** The research skills require `branch@commit` provenance, so every hash cited in reports, ledgers, `STATUS.json` files, reviews and the version index would point to commits that no longer exist in the rewritten history.
  - It needs a force-push of all 12 branches and the 37 archive tags, which the project's rules forbid without the owner's explicit decision.
  - GitHub keeps the old objects reachable through PR refs (`refs/pull/1`, `refs/pull/2`) and caches until GitHub support runs garbage collection. The repository's displayed size may not fall without that request.
  - Every existing clone, including Mr. Zhou's machines, Codex's checkouts and the cloud environment's cache, must be thrown away and recloned. Any push from an old clone would bring the old history back.
  - Any of Mr. Zhou's other tools that store commit hashes would break.
- **Who acts:** only Mr. Zhou can decide it. Then one person would run the rewrite and force-push while every agent stops, everyone would reclone, and GitHub support would be asked to garbage-collect.
- **Reversible:** only from a complete mirror backup made beforehand (`git clone --mirror`, about 10 GiB of storage somewhere), restored by another force-push. Without that backup it cannot be undone. Even with it, clones made between the rewrite and the restore would diverge.

## 7. What was not measured, and why

- **A real depth-1 clone of the whole default branch.** It needs about 21 GiB; the session had 8.8 GB free and a 2 GB limit on test use. Its size is computed from this clone's pack (9.98 GiB packed for the tip's blobs) and the tip's blob table (11.12 GiB checkout). It was not downloaded.
- **Packed sizes as GitHub stores them.** Packed figures are this session's pack (`%(objectsize:disk)`). GitHub's own pack may delta differently. GitHub's total (10,530,576 KiB) agrees with the local pack (10,533,916 KiB) to within 0.1 %. Packed sizes of the 79 blobs found only beyond the shallow boundary are counted at their uncompressed size, 2.9 MB in all.
- **Growth before the shallow boundary of this clone.** No gap here: the blob-less mirror holds every commit, and every blob size is known (section "How").
- **Growth by push time.** The table uses committer dates. A commit made on one day and pushed later, or rebased, counts on its earliest committer date.
- **Whether weights were actually read.** No access log exists. Section 5 measures only whether code in the repository names them for loading.
- **Pushing from a shallow sparse clone, and making the cloud environment's first clone sparse.** Neither was tested: pushing was out of scope, and the environment's first clone is set outside the repository.
- **Git LFS quota and cost for this account, and GitHub's current release-asset limits.** Not queried; the limits quoted are GitHub's published ones as I know them.
- **Duplicates inside the `.gz`, `.npz` and `.pt` files**, such as the same tensor stored in two files under different compression. Not examined; only whole-blob identity and zip entries against plain files were checked.

Scratch material (the blob-less mirror, size tables and scripts) was kept only in the session scratchpad and deleted at the end.
