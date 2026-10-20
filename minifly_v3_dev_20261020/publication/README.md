# MiniFly v3 publication

Three engineering cycles completed: **NO_ADVANCE**. Native taught association/retention (E1) was demonstrated; unseen-addition improvement (E3) was not. No formal pilot or confirmation ran.

- [Result report](../REPORT.md)
- [Revised v3 design](../../minifly_v3_design_20261020/DESIGN.md)
- [Originally delivered design memo](../cycles/cycle1/DESIGN_BEFORE.md)
- [Training accounting](../TRAINING_LEDGER.json): 207,450 full technical record applications, including fixtures and parent controls; zero scientific records.
- [Audit](../FINAL_AUDIT.json) and [package identity](../PACKAGE.json)

The original 210,655,877-byte ZIP exceeds GitHub's ordinary file limit. Ordered parts below preserve it exactly; source, receipts, checkpoints and reports are also published as individual files. From this directory reconstruct and verify:

```sh
cat RESULT_BUNDLE.zip.part* > ../RESULT_BUNDLE.zip
shasum -a 256 ../RESULT_BUNDLE.zip
```

Expected SHA-256: `6b3b1e60838b740094d11e490eecdb2c410972cbd71936415d21c4852803f2f0`. See ARCHIVE_PARTS.json for individual part hashes. The original local ZIP is preserved unchanged. Scratch and compiled caches are excluded.
