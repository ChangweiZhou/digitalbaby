# E24 host check, second wave (12 November 2026): D-ans 93683 and 116089 equal E20-B's G0 births byte for byte

**Instruction:** the coordinator's instruction under brief revision 4 (17a5fa35). The backprop baselines are taken on this host, compared with `exp45/e20b/e20_dev_G0_<seed>` (review 9.101 B).

**Run:**
- Sources: byte-identical to the first wave (f4c6aa8c; unchanged at 17a5fa35; `work2_sources_sha256.txt`).
- Host: the same four-core cloud Xeon, Python 3.11.18, torch 2.15.0+cpu, one thread per process.
- Work folder: fresh, with an `exp45` holding only `BIRTH_MANIFEST.json`.
- Command: `E16_ARCH=bank2 E16_BOOKDECAY=off python3 exp16_integrated.py dev <seed>`. The two ran together from 23:59:11 UTC, with no other process (`environ.txt`, `wall.txt`).

| Seed | Exit | New receipt written | Sealed scores, six checkpoints | Losses | Weights, 24 tensors | Hard mean by checkpoint (k = 12, add89 and sub89) | CPU |
|---|---|---|---|---|---|---|---:|
| 93683 | 0 at 00:54:49 | 00:54:49, `complete` true | identical to G0 | identical | 24 of 24 bit-equal | 15.8, 50.5, 77.7, 90.0, 91.0, **92.8** | 3,301 s |
| 116089 | 0 at 00:53:47 | 00:53:46, `complete` true | identical to G0 | identical | 24 of 24 bit-equal | 21.8, 51.2, 84.3, 89.2, 91.0, **92.8** | 3,238 s |

- The weight file hashes differ from G0's because the two loops write different file formats. The tensors are bit-equal.
- The first wave's D-ans 73129 rerun also equals `e20_dev_G0_73129` in all six sealed checkpoints and all 24 tensors (`compare_G0_73129_with_first_wave_rerun.json`).

**D-ans baselines on this host:** 93.2 / 92.8 / 92.8.

Nothing else was run. The nine new births wait for the coordinator's release.
