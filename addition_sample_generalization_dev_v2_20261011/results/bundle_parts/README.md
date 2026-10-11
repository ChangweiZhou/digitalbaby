# Reconstruct the complete science bundle

All 492 accepted DEV/CONFIRM receipts and their audit certificates, frozen sources, reports and the operations-only recovery are inside this archive. Raw receipts are not duplicated as loose Git blobs.

Run in this directory:

```sh
cat RESULT_BUNDLE.zip.part* > RESULT_BUNDLE.zip
shasum -a 256 RESULT_BUNDLE.zip
unzip RESULT_BUNDLE.zip
```

Expected archive SHA-256: `b91e264b7ed3a469d728c55683900b239467f4e7e5ffa8174f3ba3a7a424273f`. The separate technical `../../RESULT_BUNDLE.zip` records the three qualification cycles; its contents are not science.
