# Cycle 1 · Round 20 source bridge

Refine. Require the original compressor, its world generator, and a source-bound historical receipt before replay. Keep the recovered GLN from a different round outside the parent gate.

Audit. Search the workspace, downloads, available attachments and parallel checkout for cmr.py and the Round 20 result directory. GitHub commit searches for Round 20 and cmr.py returned no result. The memo was present.

Test. Exercise admission on the missing-source state and on synthetic hash-match and hash-mismatch receipts. Even a hash match requires independent replay.

Result. The admission canaries passed. The real bridge stopped at missing parent material. No historical source was reconstructed or trained, and source-specific timing and shuffle behavior remain unverified.
