# MiniFly sparse store capacity test

Four paired seeds per cell. The recovered V40 SUPER arm uses the V39-derived stream,
lag encoder and local error update. Only layer dimensions and active counts vary within
a load. All 12 streams were audited for absence of alias-2 contexts in training windows.
The alias retains the same taught key and label. This is E2 presentation transfer;
a canonical key table could solve it too.

| Facts | Units | Active | Seeds | Old after D | Unseen after D | Old after clutter | Unseen after clutter | One-shot unseen | Revision @8 unseen | Store MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4096 | 205 | 4 | 68.8% | 46.9% | 32.8% | 14.1% | 99.8% | 100.0% | 8 |
| 128 | 16384 | 205 | 4 | 95.3% | 95.3% | 85.9% | 90.6% | 99.4% | 100.0% | 32 |
| 128 | 16384 | 410 | 4 | 94.5% | 96.9% | 91.4% | 93.8% | 100.0% | 100.0% | 32 |
| 256 | 16384 | 205 | 4 | 87.9% | 84.4% | 77.0% | 73.4% | 98.8% | 100.0% | 32 |
| 512 | 16384 | 205 | 4 | 50.2% | 36.3% | 44.9% | 32.0% | 95.9% | 99.2% | 32 |
| 512 | 65536 | 205 | 4 | 91.6% | 80.1% | 89.3% | 83.6% | 97.0% | 100.0% | 128 |
| 512 | 65536 | 410 | 4 | 93.9% | 90.2% | 93.6% | 92.6% | 99.1% | 100.0% | 128 |

## Paired effects on retention after wave D

Intervals are descriptive paired 95% t intervals across four seeds.
- (128, 4096, 205) → (128, 16384, 205), old: +26.6 pp [-2.3, +55.4].
- (128, 4096, 205) → (128, 16384, 205), unseen: +48.4 pp [+33.5, +63.4].
- (512, 16384, 205) → (512, 65536, 205), old: +41.4 pp [+34.4, +48.4].
- (512, 16384, 205) → (512, 65536, 205), unseen: +43.8 pp [+37.7, +49.8].
- (128, 16384, 205) → (128, 16384, 410), old: -0.8 pp [-3.3, +1.7].
- (128, 16384, 205) → (128, 16384, 410), unseen: +1.6 pp [-3.4, +6.5].
- (512, 65536, 205) → (512, 65536, 410), old: +2.3 pp [+0.1, +4.6].
- (512, 65536, 205) → (512, 65536, 410), unseen: +10.2 pp [-0.1, +20.4].

The store uses the recovered float64 arithmetic. Code arrays are additional, and
the encoder scans all units for top-k selection. These are workload-specific capacity
measurements, with no byte-efficiency or V181-fit claim.
