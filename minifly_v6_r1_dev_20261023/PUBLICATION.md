# Results-only GitHub publication

The user authorized: “Push the results, not the raw data.”

Published: the original V6-R1 design, implementation, three-cycle reviews and technical receipts, aggregate metrics/curves/diagnostics, final audit, report, revision, proposed next design and source/provenance hashes.

Not published: raw item-level predictions or margins, checkpoints, calibration arrays, dataset/stream manifests, per-history/progress files, runtime logs, caches, the full local result archive or its split parts. These files remain unchanged locally. The proposed next design is not executed or adopted.

`PUBLICATION_MANIFEST.json` inventories the published files. `SOURCE_LOCK.json` is the original scientific dependency lock and can refer to deliberately omitted local files; it is not an assertion that those files are on GitHub. The archived full-run manifest and package metadata remain local. Artifact-audit scripts require the omitted local evidence; no training or trajectories were rerun for publication.

The scientific result is unchanged: 34,816 external record applications, 393,472 observed native byte/EOS events, eight histories in two engineering worlds, and no demonstrated unseen-addition gain. All API/provenance/telemetry limitations remain disclosed in the report and revision.
