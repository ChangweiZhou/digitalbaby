# addcheck3.py: removals from addcheck2.py variant carry (D = 100 unless stated; 240,000 lines; 5 orders a seed)

Values: smallest / median / largest over the 5 orders of a seed, in %. Never shown = the pairs held out (2,000 of 10,000; 20 of 100 for D = 10).
Curve: median over the 5 orders of the never-shown whole score after 60k / 120k / 240k lines.

| Variant | Seed | Shown whole | Never shown whole | Never shown tens | Never shown units | Curve (never shown whole) | Yes while learning |
|---|---|---|---|---|---|---|---|
| base (carry cell, holding cells laid down) | 73129 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| base (carry cell, holding cells laid down) | 93683 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| base (carry cell, holding cells laid down) | 116089 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| (a) carry=learned | 73129 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| (a) carry=learned | 93683 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| (a) carry=learned | 116089 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| (b) hold=pool, all 741 pairs | 73129 | 8.4 / 9.0 / 10.5 | 6.5 / 7.1 / 8.5 | 31.9 / 33.0 / 33.7 | 20.9 / 22.2 / 23.2 | 2.3 / 5.0 / 7.1 | 36.9 |
| (b) hold=pool, all 741 pairs | 93683 | 8.9 / 9.3 / 9.8 | 6.2 / 6.8 / 7.1 | 31.3 / 31.7 / 32.6 | 20.2 / 20.9 / 22.1 | 2.9 / 4.6 / 6.8 | 37.0 |
| (b) hold=pool, all 741 pairs | 116089 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| (b) hold=pool, 741 pairs + cpairs=2000 | 73129 | 7.7 / 9.6 / 11.0 | 5.3 / 6.8 / 9.4 | 31.7 / 33.0 / 33.2 | 19.5 / 22.2 / 25.5 | 2.5 / 4.3 / 6.8 | 36.9 |
| (b) hold=pool, 741 pairs + cpairs=2000 | 93683 | 8.1 / 9.1 / 9.7 | 5.7 / 6.2 / 7.6 | 30.3 / 32.6 / 35.4 | 19.4 / 20.1 / 22.5 | 2.9 / 4.7 / 6.2 | 36.9 |
| (b) hold=pool, 741 pairs + cpairs=2000 | 116089 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| (b) hold=pool, D=10, 60,000 lines | 73129 | 82.5 / 85.0 / 88.8 | 0.0 / 0.0 / 0.0 | 83.3 / 83.3 / 83.3 | 10.0 / 10.0 / 10.0 | 0.0 | 60.7 |
| (b) hold=pool, D=10, 60,000 lines | 93683 | 83.8 / 87.5 / 93.8 | 0.0 / 0.0 / 5.0 | 100.0 / 100.0 / 100.0 | 0.0 / 0.0 / 5.0 | 0.0 | 61.2 |
| (b) hold=pool, D=10, 60,000 lines | 116089 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| (c) strength=0.1, eq=0 | 73129 | 32.2 / 32.6 / 33.4 | 28.4 / 31.5 / 31.8 | 57.3 / 58.8 / 60.9 | 51.5 / 53.7 / 55.0 | 18.0 / 26.7 / 31.5 | 45.1 |
| (c) strength=0.1, eq=0 | 93683 | 31.0 / 31.7 / 34.7 | 27.6 / 29.9 / 32.8 | 57.7 / 61.1 / 63.9 | 49.9 / 52.0 / 56.1 | 18.9 / 26.6 / 29.9 | 45.2 |
| (c) strength=0.1, eq=0 | 116089 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| (c) strength=0.1, eq=1 | 73129 | 26.6 / 28.8 / 30.4 | 25.2 / 27.0 / 29.9 | 69.2 / 71.7 / 75.5 | 37.6 / 40.5 / 41.2 | 40.9 / 70.1 / 27.0 | 48.9 |
| (c) strength=0.1, eq=1 | 93683 | 27.9 / 29.1 / 31.8 | 27.3 / 27.5 / 30.9 | 67.8 / 69.6 / 75.0 | 36.9 / 38.5 / 41.5 | 39.7 / 68.5 / 27.5 | 49.0 |
| (c) strength=0.1, eq=1 | 116089 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| (2) M=8 output part | 73129 | 88.5 / 90.2 / 93.2 | 87.9 / 89.5 / 92.1 | 97.5 / 97.5 / 97.5 | 90.2 / 91.1 / 94.0 | 41.6 / 67.5 / 89.5 | 49.0 |
| (2) M=8 output part | 93683 | 90.1 / 90.9 / 92.6 | 89.6 / 90.0 / 92.0 | 97.5 / 97.5 / 100.0 | 91.4 / 91.8 / 92.7 | 44.9 / 67.8 / 90.0 | 49.3 |
| (2) M=8 output part | 116089 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

NOT_RUN: base and learned on every seed (the batch file `batch_add3.txt` had a trailing space on the base lines, so `xargs -L1`
joined each base line to the next learned line; base stopped with a ValueError on its arguments and learned never started);
every variant on seed 116089 (not started before the stop instruction of 16:13 UTC).
