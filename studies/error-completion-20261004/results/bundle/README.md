# Lossless public scientific supplement

This folder contains all **8,453 reviewed final scientific files** as a lossless
tar.xz archive split into 89 small binary chunks. It includes the 64 public world
receipts, their complete index, and exact reviewed snapshots of the final report,
analysis, scientific verification summary, and publication manifest.

The decoded files total 435,857,266 bytes. The compressed archive is 34,716,904
bytes. Each chunk, each file-index part, and every decoded file has a SHA256
identity in `BUNDLE_MANIFEST.json` and the readable `file-index-*.json` parts.
The manifest also identifies the restore program and this document. The manifest
itself is identified by the [final publication manifest](../final/PUBLICATION_MANIFEST.json).

## Download and restore

Obtain this repository branch using GitHub's **Code → Download ZIP** and unpack
it, or clone `study/error-completion-20261004`. Keep this whole bundle folder,
including its `chunks` directory and all file-index parts, together.

From the repository root, with Python 3.9 or later:

```sh
python3 studies/error-completion-20261004/results/bundle/restore_bundle.py --verify-only
python3 studies/error-completion-20261004/results/bundle/restore_bundle.py --output error-restored
```

`error-restored` must not already exist. The utility performs a complete validation
before creating that directory, then validates every file again while restoring
its exact original relative path under `error-restored/studies/error-completion-20261004/`.
It rejects missing, altered, duplicate, unsafe, or nonregular archive members.
It uses only Python's standard library, performs no network requests, and never
overwrites an existing destination. On a write failure it may leave an incomplete
new output directory; a successful run prints `"verified": true` with all 8,453 files.

For a standard archive tool, concatenate the chunks in their numbered order to
form `scientific-files.tar.xz`, then extract into a new directory. This alone does
not perform the complete per-file verification; use the supplied utility to check
the original chunks and all decoded bytes.

## Reading and interpretation

The live [report](../../FINAL_REPORT.md), [analysis](../final/analysis.json), and
[scientific audit summary](../../audits/FINAL_ANALYSIS_SUMMARY.json) remain directly
readable on GitHub. The raw receipt-relative paths in the analysis and archived
report resolve after restoration. The archived report and publication manifest
are exact reviewed snapshots; the live versions add these delivery directions.
Their scientific numbers, interpretations, and limitations are unchanged.

This is a **supplement**, not a replacement for the initial 397-file source and
qualification release. That release and its [README](../../README.md),
[protocol](../../protocol/SCIENTIFIC_LOCK.json), [NOTICE](../../NOTICE.md), and
[license](../../COPYING) remain ordinary files in the repository. Use the restored
scientific supplement alongside that checkout. Public receipt headers/manifests
are explicitly labeled projections; they do not reproduce excluded private
execution identities and are not inputs to validators requiring those identities.

No private checkpoints, administrative records, approval/session identities, or
private source/control paths are included. The bundle changes delivery only, not
the fixed 64-world sample, scientific settings, analysis, or conclusions.

SPDX-License-Identifier: GPL-3.0-or-later. Inherited attribution is preserved.
