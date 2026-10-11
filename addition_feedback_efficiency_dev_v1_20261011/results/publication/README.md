# Complete result archive

GitHub limits individual Git blobs to 100 MB. The accepted 354,009,745-byte ZIP is split into five parts no larger than 75 MiB. All 456 full scientific receipts, audits, manifests, compact metrics, source snapshots and reports are in this archive. They are not duplicated as loose receipt files in Git.

From this directory, reconstruct without changing the archive:

```sh
cat RESULT_BUNDLE.zip.part* > RESULT_BUNDLE.zip
unzip RESULT_BUNDLE.zip
```

The reports and compact metrics are also visible directly under results/. BUNDLE_MANIFEST.json records the accepted archive digest from PACKAGE.json; no science was rerun for publication.
