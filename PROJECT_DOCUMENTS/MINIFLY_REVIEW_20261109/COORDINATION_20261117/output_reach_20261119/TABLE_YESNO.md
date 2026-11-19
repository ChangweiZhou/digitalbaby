# The output end learning online from yes/no (yesno.py, yesno2.py): share right, %

Cell: the last window of 4,096 bytes for the best setting of the family (the setting in brackets: eps, eta or kappa), then the bytes after which a window first reached the share of the most frequent byte of that window (f) and ref_1 of that window (r1); "-" never.

## 65,536 bytes

| input | rule | seed 73129 | seed 93683 | seed 116089 |
|---|---|---|---|---|
| present byte (one-hot) | R1 | 0.5 (eps=0 eta=0.0625); f -, r1 - | 0.6 (eps=0 eta=0.0625); f -, r1 - | 0.7 (eps=0 eta=0.0625); f -, r1 - |
| present byte (one-hot) | R2 | 0.5 (eps=0.0625 eta=0.0625); f -, r1 - | 0.6 (eps=0.015625 eta=1); f -, r1 - | 0.6 (eps=0.015625 eta=1); f -, r1 - |
| present byte (one-hot) | R3 | 0.7 (eps=0 eta=0.0625); f -, r1 - | 0.6 (eps=0.0625 eta=0.0625); f -, r1 - | 0.7 (eps=0.015625 eta=1); f -, r1 - |
| present byte (one-hot) | A | 7.2 (eps=0 kappa=1/8); f 40960, r1 - | 7.3 (eps=0.0625 kappa=1/8); f 61440, r1 - | 7.1 (eps=0.0625 kappa=1/8); f 49152, r1 - |
| present byte (one-hot) | B | 9.2 (eps=0.0625); f 16384, r1 - | 9.2 (eps=0.0625); f 4096, r1 32768 | 8.9 (eps=0.0625); f 8192, r1 32768 |
| present byte (one-hot) | D | 8.4 (eps=0 eta=0.0625); f 4096, r1 - | 10.1 (eps=0 eta=0.0625); f 4096, r1 - | 9.6 (eps=0 eta=0.0625); f 4096, r1 - |
| present + previous byte | R1 | 0.6 (eps=0 eta=0.0625); f -, r1 - | 0.4 (eps=0 eta=0.0625); f -, r1 - | 0.4 (eps=0 eta=0.0625); f -, r1 - |
| present + previous byte | R2 | 0.9 (eps=0.015625 eta=0.0625); f -, r1 - | 0.6 (eps=0.015625 eta=0.25); f -, r1 - | 0.7 (eps=0.0625 eta=0.0625); f -, r1 - |
| present + previous byte | R3 | 1.0 (eps=0.015625 eta=0.0625); f -, r1 - | 0.7 (eps=0.015625 eta=0.25); f -, r1 - | 0.7 (eps=0 eta=0.0625); f -, r1 - |
| present + previous byte | A | 8.7 (eps=0 kappa=1/8); f 4096, r1 - | 8.3 (eps=0 kappa=1/8); f 8192, r1 - | 8.0 (eps=0 kappa=1/8); f 4096, r1 - |
| present + previous byte | B | 6.7 (eps=0.0625); f 24576, r1 - | 10.9 (eps=0.0625); f 4096, r1 16384 | 10.7 (eps=0.0625); f 8192, r1 28672 |
| present + previous byte | D | 11.6 (eps=0 eta=0.0625); f 4096, r1 4096 | 12.5 (eps=0 eta=0.0625); f 4096, r1 4096 | 12.6 (eps=0 eta=0.0625); f 4096, r1 4096 |
| last three bytes | A | 12.1 (eps=0 kappa=1/8); f 8192, r1 8192 | 13.7 (eps=0.0625 kappa=1/8); f 8192, r1 8192 | 12.2 (eps=0.0625 kappa=1/8); f 4096, r1 8192 |
| last three bytes | B | 14.2 (eps=0.0625); f 8192, r1 12288 | 13.8 (eps=0.0625); f 4096, r1 16384 | 16.2 (eps=0.0625); f 8192, r1 28672 |
| U cells near own driver | R1 | 0.5 (eps=0 eta=0.0625); f -, r1 - | 0.3 (eps=0 eta=0.0625); f -, r1 - | 0.4 (eps=0 eta=0.0625); f -, r1 - |
| U cells near own driver | R2 | 0.5 (eps=0.0625 eta=1); f -, r1 - | 0.4 (eps=0.0625 eta=1); f -, r1 - | 0.4 (eps=0.0625 eta=0.0625); f -, r1 - |
| U cells near own driver | R3 | 0.8 (eps=0.0625 eta=0.0625); f -, r1 - | 0.9 (eps=0.015625 eta=0.0625); f -, r1 - | 0.8 (eps=0.0625 eta=0.25); f -, r1 - |
| U cells near own driver | A | 5.2 (eps=0 kappa=1/46); f -, r1 - | 5.5 (eps=0 kappa=1/46); f -, r1 - | 6.5 (eps=0 kappa=1/46); f 8192, r1 - |
| U cells near own driver | B | 1.6 (eps=0.0625); f -, r1 - | 2.1 (eps=0.0625); f -, r1 - | 4.0 (eps=0.0625); f -, r1 - |
| U cells near own driver | D | 4.5 (eps=0 eta=0.0625); f -, r1 - | 5.1 (eps=0 eta=0.0625); f -, r1 - | 6.6 (eps=0 eta=0.0625); f 28672, r1 - |
| M cells near own driver | R1 | 0.3 (eps=0 eta=0.0625); f -, r1 - | 0.3 (eps=0 eta=0.0625); f -, r1 - | 0.3 (eps=0 eta=0.0625); f -, r1 - |
| M cells near own driver | R2 | 0.4 (eps=0.015625 eta=0.25); f -, r1 - | 0.4 (eps=0.0625 eta=0.25); f -, r1 - | 0.5 (eps=0.015625 eta=0.25); f -, r1 - |
| M cells near own driver | R3 | 0.8 (eps=0 eta=0.0625); f -, r1 - | 0.7 (eps=0 eta=0.0625); f -, r1 - | 0.9 (eps=0.015625 eta=1); f -, r1 - |
| M cells near own driver | A | 4.9 (eps=0 kappa=1/46); f -, r1 - | 5.8 (eps=0 kappa=1/46); f 32768, r1 - | 4.3 (eps=0.0625 kappa=1/46); f -, r1 - |
| M cells near own driver | B | 2.8 (eps=0.0625); f -, r1 - | 3.9 (eps=0.0625); f -, r1 - | 5.6 (eps=0.0625); f -, r1 - |
| M cells near own driver | D | 5.1 (eps=0 eta=0.0625); f -, r1 - | 5.2 (eps=0 eta=0.0625); f -, r1 - | 4.9 (eps=0 eta=0.0625); f -, r1 - |
| reference | most frequent byte of the window | 6.3 | 6.3 | 6.3 |
| reference | end mark (byte 10) | 6.3 | 6.3 | 6.3 |
| reference | ref_1 | 10.0 | 10.6 | 10.8 |
| reference | ref_2 | 11.9 | 12.5 | 12.0 |
| reference | ref_3 | 25.6 | 25.7 | 26.0 |

## 262,144 bytes

| input | rule | seed 73129 | seed 93683 | seed 116089 |
|---|---|---|---|---|
| present byte (one-hot) | A | 8.5 (eps=0 kappa=1/8); f 40960, r1 - | 9.1 (eps=0 kappa=1/8); f 28672, r1 - | 9.2 (eps=0 kappa=1/8); f 69632, r1 - |
| present byte (one-hot) | B | 8.3 (eps=0.0625); f 16384, r1 - | 8.7 (eps=0.0625); f 4096, r1 32768 | 8.9 (eps=0.0625); f 8192, r1 32768 |
| last three bytes | A | 12.4 (eps=0 kappa=1/8); f 8192, r1 8192 | 13.5 (eps=0.0625 kappa=1/8); f 8192, r1 8192 | 11.8 (eps=0.0625 kappa=1/8); f 4096, r1 8192 |
| last three bytes | B | 17.7 (eps=0.0625); f 8192, r1 12288 | 18.5 (eps=0.0625); f 4096, r1 16384 | 20.4 (eps=0.0625); f 8192, r1 28672 |
| U cells near own driver | A | 5.5 (eps=0 kappa=1/46); f -, r1 - | 5.5 (eps=0 kappa=1/46); f -, r1 - | 6.5 (eps=0 kappa=1/46); f 8192, r1 - |
| U cells near own driver | B | 2.2 (eps=0.0625); f -, r1 - | 2.1 (eps=0.0625); f -, r1 - | 4.1 (eps=0.0625); f -, r1 - |
| M cells near own driver | A | 5.1 (eps=0 kappa=1/46); f -, r1 - | 5.9 (eps=0 kappa=1/46); f 32768, r1 - | 4.4 (eps=0 kappa=1/46); f -, r1 - |
| M cells near own driver | B | 3.1 (eps=0.0625); f -, r1 - | 3.8 (eps=0.0625); f -, r1 - | 4.4 (eps=0.0625); f -, r1 - |
| reference | most frequent byte of the window | 6.2 | 6.2 | 6.2 |
| reference | end mark (byte 10) | 6.2 | 6.2 | 6.2 |
| reference | ref_1 | 10.5 | 10.7 | 10.9 |
| reference | ref_2 | 13.5 | 12.9 | 13.3 |
| reference | ref_3 | 27.4 | 27.0 | 28.2 |

