# V148 summary: c2_dev_passes (buster)

| Taught KB | Pass | MiniFly held bits/byte | T | held acc | taught first 5 KB | taught last 5 KB | never-seen mass | renorm-seen held (diag.) | unigram | bigram | PPM-2 | PPM-3 | PPM-4 | PPM-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 8.006 | 0.02 | 0.000 | – | – | 0.996 | – | 8.006 | 8.006 | 8.006 | 8.006 | 8.006 | 8.006 |
| 2.5 | 1 | 7.035 | 0.77 | 0.107 | 6.080 | 6.080 | 0.704 | – | 4.537 | 3.666 | 3.456 | 3.454 | 3.458 | 3.468 |
| 5 | 1 | 7.001 | 0.77 | 0.103 | 6.218 | 6.217 | 0.687 | – | 4.496 | 3.592 | 3.291 | 3.227 | 3.227 | 3.255 |
| 10 | 1 | 6.945 | 0.77 | 0.117 | 6.494 | 6.110 | 0.659 | – | 4.468 | 3.539 | 3.129 | 3.006 | 2.993 | 3.018 |
| 10 | 2 | 6.975 | 0.77 | 0.115 | 6.248 | 5.827 | 0.647 | – | 4.468 | 3.539 | 3.129 | 3.006 | 2.993 | 3.018 |
| 10 | 4 | 7.029 | 1.00 | 0.116 | 6.162 | 5.820 | 0.672 | – | 4.468 | 3.539 | 3.129 | 3.006 | 2.993 | 3.018 |

Own generations: 10; stopped at own EOS: 5; mean length 50.1 bytes; printable-ASCII fraction 0.675.

First five (prefix → own continuation | true continuation):
* `'  They are'` → `'Ò «zs ¬hkn\x7ftzn\x93eÓ\x1f.¬\x1fó h\x93 g¾¬¬÷kk »R« t¬zóRma ek\x1c '` | `' no worse; they\'re not so bad!"'`
* `'That is wh'` → `'k.\x1fígít EzÑoheR¬kk ÷t\x93.E "Aet«, R ekkHeÑtú :nt end'` | `"at Farmer Brown's boy said when he found that Bust"`
* `'stolen the'` → `'zkÓa k¬Rhin ¬»\x06fah¬\x94kglto« (« ñ¬zÑ\x93k «s «atk«÷n«zk'` | `' berries he had worked so hard to pick and then ha'`
* `'with the p'` → `'aÑ¬.eö« soRkó¬s\x1f¬ tk\x06 zs  e Rkizthx  Râ lRkaÏkhR\x94'` | `"ail. You see, Farmer Brown's boy is learning to be"`
* `'of a philo'` → `'kk\x8f\x1feó«åizz\x87zkÑ,.  H«khe\x8denï \x93å\x93¬ÚRRÏ \x1f> Ò aha*¬« '` | `'sopher, one of those people who accept bad things '`
