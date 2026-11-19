# Next-byte reader on the cells (reach.py): share right on the scored quarter, %

U: rule=fixed, no control. M: rule=fixed gain=2 rho=1 gmax=4. Each cell: R1 / R2 / rest / all (n of the scored positions in the first row of each seed).

## 16,384 bytes (fitted on the first three quarters, scored on the last quarter)

- seed 73129: (a) 495 cells (256 drivers and their neighbourhoods), distance from the input lines 7: 2, 8: 51, 9: 175, 10: 209, 11: 58; (b) 1984 cells
- seed 93683: (a) 519 cells (256 drivers and their neighbourhoods), distance from the input lines 7: 7, 8: 88, 9: 208, 10: 201, 11: 15; (b) 1984 cells
- seed 116089: (a) 495 cells (256 drivers and their neighbourhoods), distance from the input lines 8: 41, 9: 154, 10: 221, 11: 79; (b) 1984 cells

| reader | seed 73129 | seed 93683 | seed 116089 |
|---|---|---|---|
| U, (a) near the output lines | 14.5 / 10.2 / 12.4 / **12.4** (n 256/256/3584) | 14.0 / 11.2 / 12.1 / **12.2** (n 258/259/3579) | 9.8 / 6.8 / 13.6 / **12.9** (n 264/264/3568) |
| U, (b) every non-feedback cell | 12.5 / 9.4 / 13.6 / **13.3** | 14.7 / 10.0 / 14.0 / **13.8** | 9.8 / 9.5 / 14.0 / **13.4** |
| M, (a) near the output lines | 10.2 / 5.9 / 16.2 / **15.1** | 8.5 / 10.4 / 12.0 / **11.7** | 10.2 / 6.1 / 14.2 / **13.4** |
| M, (b) every non-feedback cell | 15.6 / 9.0 / 17.4 / **16.8** | 14.7 / 6.9 / 14.4 / **14.0** | 10.2 / 11.0 / 15.2 / **14.6** |
| reader on the present byte (one-hot) | 7.8 / 0.0 / 11.0 / **10.1** | 10.9 / 0.0 / 11.6 / **10.8** | 10.6 / 0.0 / 12.1 / **11.3** |
| reader on present + previous byte (two one-hots) | 8.6 / 9.0 / 12.2 / **11.8** | 11.2 / 11.2 / 12.8 / **12.6** | 11.7 / 11.4 / 14.1 / **13.7** |
| reader on the last three bytes (three one-hots) | 34.0 / 34.4 / 26.8 / **27.8** | 33.7 / 34.4 / 26.8 / **27.7** | 33.0 / 34.1 / 28.3 / **29.0** |
| ref_mode (scorer) | 0.0 / 0.0 / 0.0 / **0.0** | 0.0 / 0.0 / 0.0 / **0.0** | 0.0 / 0.0 / 0.0 / **0.0** |
| ref_1 (scorer) | 7.4 / 0.0 / 10.6 / **9.8** | 8.1 / 0.0 / 11.3 / **10.4** | 10.6 / 0.0 / 12.0 / **11.2** |
| ref_2 (scorer) | 9.8 / 7.8 / 11.7 / **11.4** | 11.2 / 9.3 / 11.7 / **11.5** | 8.7 / 9.1 / 12.9 / **12.4** |
| ref_3 (scorer) | 26.6 / 27.0 / 19.6 / **20.5** | 26.7 / 27.8 / 19.3 / **20.3** | 22.0 / 25.4 / 20.9 / **21.3** |

## 65,536 bytes (fitted on the first three quarters, scored on the last quarter)

- seed 73129: (a) 495 cells (256 drivers and their neighbourhoods), distance from the input lines 7: 2, 8: 51, 9: 175, 10: 209, 11: 58; (b) 1984 cells
- seed 93683: (a) 519 cells (256 drivers and their neighbourhoods), distance from the input lines 7: 7, 8: 88, 9: 208, 10: 201, 11: 15; (b) 1984 cells
- seed 116089: (a) 495 cells (256 drivers and their neighbourhoods), distance from the input lines 8: 41, 9: 154, 10: 221, 11: 79; (b) 1984 cells

| reader | seed 73129 | seed 93683 | seed 116089 |
|---|---|---|---|
| U, (a) near the output lines | 19.6 / 16.5 / 14.2 / **14.7** (n 1017/1017/14350) | 13.2 / 10.7 / 12.9 / **12.8** (n 1030/1030/14324) | 15.0 / 11.2 / 14.0 / **13.9** (n 1023/1023/14338) |
| U, (b) every non-feedback cell | 20.1 / 17.6 / 14.1 / **14.7** | 15.0 / 11.0 / 14.4 / **14.3** | 13.8 / 11.5 / 14.2 / **14.1** |
| M, (a) near the output lines | 10.9 / 6.7 / 16.2 / **15.3** | 8.3 / 7.4 / 12.9 / **12.3** | 10.3 / 5.1 / 13.4 / **12.7** |
| M, (b) every non-feedback cell | 17.3 / 10.0 / 17.0 / **16.6** | 14.5 / 8.4 / 15.5 / **15.0** | 15.3 / 7.8 / 15.0 / **14.6** |
| reader on the present byte (one-hot) | 10.9 / 0.0 / 11.3 / **10.6** | 9.8 / 0.0 / 11.1 / **10.4** | 11.9 / 0.0 / 11.5 / **10.8** |
| reader on present + previous byte (two one-hots) | 12.3 / 10.9 / 13.1 / **13.0** | 9.8 / 9.0 / 12.9 / **12.5** | 11.9 / 12.2 / 12.8 / **12.7** |
| reader on the last three bytes (three one-hots) | 33.8 / 33.4 / 26.5 / **27.4** | 35.3 / 31.2 / 26.9 / **27.7** | 34.9 / 33.8 / 26.8 / **27.7** |
| ref_mode (scorer) | 0.0 / 0.0 / 0.0 / **0.0** | 0.0 / 0.0 / 0.0 / **0.0** | 0.0 / 0.0 / 0.0 / **0.0** |
| ref_1 (scorer) | 12.4 / 0.0 / 11.2 / **10.5** | 10.0 / 0.0 / 11.4 / **10.6** | 11.9 / 0.0 / 11.3 / **10.6** |
| ref_2 (scorer) | 13.2 / 10.7 / 12.4 / **12.3** | 12.5 / 9.4 / 12.3 / **12.1** | 9.4 / 9.3 / 12.6 / **12.2** |
| ref_3 (scorer) | 33.4 / 33.6 / 23.9 / **25.1** | 34.7 / 30.6 / 24.3 / **25.3** | 34.6 / 33.6 / 24.1 / **25.4** |

