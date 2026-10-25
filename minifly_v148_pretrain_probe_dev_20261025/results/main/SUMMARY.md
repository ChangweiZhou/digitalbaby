# V148 summary: main (alice)

| Taught KB | Pass | MiniFly held bits/byte | T | held acc | taught first 5 KB | taught last 5 KB | never-seen mass | renorm-seen held (diag.) | unigram | bigram | PPM-2 | PPM-3 | PPM-4 | PPM-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 8.006 | 0.02 | 0.000 | – | – | 0.996 | – | 8.006 | 8.006 | 8.006 | 8.006 | 8.006 | 8.006 |
| 5 | 1 | 7.194 | 0.77 | 0.100 | 6.425 | 6.421 | 0.660 | 5.494 | 4.669 | 3.845 | 3.545 | 3.438 | 3.477 | 3.498 |
| 10 | 1 | 7.128 | 0.77 | 0.102 | 6.744 | 6.283 | 0.629 | 5.513 | 4.655 | 3.790 | 3.391 | 3.213 | 3.244 | 3.268 |
| 20 | 1 | 7.095 | 0.77 | 0.110 | 6.808 | 6.346 | 0.618 | 5.535 | 4.636 | 3.728 | 3.224 | 2.986 | 2.989 | 3.023 |
| 40 | 1 | 7.104 | 0.88 | 0.099 | 7.028 | 6.501 | 0.634 | 5.546 | 4.625 | 3.683 | 3.119 | 2.819 | 2.778 | 2.807 |
| 40 | 2 | 7.121 | 1.00 | 0.100 | 7.013 | 6.620 | 0.645 | 5.539 | 4.625 | 3.683 | 3.119 | 2.819 | 2.778 | 2.807 |
| 40 | 4 | 7.248 | 1.30 | 0.090 | 7.130 | 6.941 | 0.660 | 5.604 | 4.625 | 3.683 | 3.119 | 2.819 | 2.778 | 2.807 |

Own generations: 50; stopped at own EOS: 21; mean length 52.5 bytes; printable-ASCII fraction 0.677.

First five (prefix → own continuation | true continuation):
* `'As soon as'` → `'«¨\x02h, R«så\x0e\x93åY h,å\x07ÈEt l« \x00hJ F}R«d« a««4'` | `' the jury had a little recovered from the shock of'`
* `'upset, and'` → `' oR½Y\x00\x00\x11 nn \x11o\x8f e\n\x8fRkz÷>Rózho\x1fk«e©(e\xad\x00  »kRkeki\x00 e'` | `' their slates and pencils had been found and hande'`
* `'them, they'` → `'åERYh\x06>hs>'` | `' set to work very diligently to write out a histor'`
* `'accident, '` → `'hådek\x00He koEk>ond Hhe ´\x04¨(\x7fkhkkéo(ka\x93«haÏå½\x7fäk¿höz'` | `'all except the Lizard, who seemed too much overcom'`
* `'anything b'` → `'$h, is ¬aó asÑ>kk ÷åRkhm»k¼, henå7ói\x00Y hhekdó'` | `'ut sit with its mouth open, gazing up into the roo'`

## Registered predictions (design section 7; operationalised in LOCK.json before the run)
* P1: NOT AS PREDICTED (worse than bigram) — minifly=7.104, bigram=3.683, ppmc3=2.819, best_ppm_order3plus=2.778
* P2: NOT AS PREDICTED (early text erased) — held_minus_taught_last5k=0.604, held_minus_taught_first5k=0.077, threshold=0.250
* P3: AS PREDICTED — minifly_improvement=0.090, ppmc2_improvement=0.425, ppmc3_improvement=0.618, ppmc4_improvement=0.699, ppmc5_improvement=0.690, range_kb=[5.0, 40.0]
