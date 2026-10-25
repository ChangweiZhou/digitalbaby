# V148 summary: c2_dev (buster)

| Taught KB | Pass | MiniFly held bits/byte | T | held acc | taught first 5 KB | taught last 5 KB | never-seen mass | renorm-seen held (diag.) | unigram | bigram | PPM-2 | PPM-3 | PPM-4 | PPM-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 8.006 | 0.02 | 0.000 | – | – | – |– | 8.006 | 8.006 | 8.006 | 8.006 | 8.006 | 8.006 |
| 2.5 | 1 | 7.035 | 0.77 | 0.107 | 6.080 | 6.080 | – |– | 4.537 | 3.666 | 3.456 | 3.454 | 3.458 | 3.468 |
| 5 | 1 | 7.001 | 0.77 | 0.103 | 6.218 | 6.217 | – |– | 4.496 | 3.592 | 3.291 | 3.227 | 3.227 | 3.255 |
| 10 | 1 | 6.945 | 0.77 | 0.117 | 6.494 | 6.110 | – |– | 4.468 | 3.539 | 3.129 | 3.006 | 2.993 | 3.018 |

Own generations: 10; stopped at own EOS: 1; mean length 60.6 bytes; printable-ASCII fraction 0.751.

First five (prefix → own continuation | true continuation):
* `'  They are'` → `'nk ìs\x93 ahs \x7fn> h EåÑ h«z\x7fo\x1f« t'` | `' no worse; they\'re not so bad!"'`
* `'That is wh'` → `'k ing kt,³hk he \x1fzh\xa0 ß\x1f¬s Ñ« Ò««kh th\x94ú« öRnk Esd '` | `"at Farmer Brown's boy said when he found that Bust"`
* `'stolen the'` → `'z\x1f,zhe\x8d ha  ¬ohkE \n oz«he¼¬ h ¬t ksz\x7fo FEkhk es\x1f e'` | `' berries he had worked so hard to pick and then ha'`
* `'with the p'` → `'k *koúzE E Ñ.YoÏååzRzheng ÷ «thened .z hathe lk«zs'` | `"ail. You see, Farmer Brown's boy is learning to be"`
* `'of a philo'` → `'kktú«o«tåkÑ \x1fad eR Eó öhk Rkh hoznk EnÑtzÑR ¬å¬ o '` | `'sopher, one of those people who accept bad things '`
