
## net13 (seeds [1, 2, 3])

Per test rule and condition: err80 / L80s / f20 / f100 / f500 / last500 (seeds 1, 2, 3; '-' = not reached)

| rule | cond | err80 | L80s | f20 | f100 | f500 | last500 | L80 (100-line) |
|---|---|---|---|---|---|---|---|---|
| X1 | fresh | 62 / 47 / 50 | 36 / 32 / 31 | 0.362 / 0.412 / 0.375 | 0.818 / 0.843 / 0.833 | 0.963 / 0.967 / 0.966 | 1.0 / 1.0 / 1.0 | 100 / 100 / 100 |
| X1 | new | 239 / 166 / 217 | 128 / 63 / 88 | 0.013 / 0.0 / 0.075 | 0.465 / 0.49 / 0.42 | 0.827 / 0.835 / 0.799 | 0.998 / 0.996 / 0.985 | 177 / 182 / 175 |
| X1 | fam | 151 / 105 / 166 | 72 / 55 / 86 | 0.075 / 0.2 / 0.087 | 0.59 / 0.685 / 0.562 | 0.9 / 0.921 / 0.89 | 0.998 / 0.997 / 0.997 | 126 / 116 / 133 |
| X1 | rnd | 70 / 57 / 66 | 41 / 34 / 41 | 0.35 / 0.388 / 0.388 | 0.725 / 0.74 / 0.775 | 0.897 / 0.916 / 0.946 | 0.993 / 1.0 / 1.0 | 112 / 107 / 103 |
| X2 | fresh | 472 / 459 / 514 | 386 / 405 / 424 | 0.175 / 0.225 / 0.312 | 0.517 / 0.605 / 0.515 | 0.726 / 0.739 / 0.711 | 0.983 / 0.986 / 0.976 | 450 / 466 / 505 |
| X2 | new | 1436 / 1520 / 1537 | 974 / - / 980 | 0.0 / 0.025 / 0.075 | 0.182 / 0.158 / 0.177 | 0.531 / 0.515 / 0.51 | 0.739 / 0.725 / 0.715 | - / - / - |
| X2 | fam | 906 / 1024 / 925 | 821 / 904 / 816 | 0.0 / 0.0 / 0.013 | 0.515 / 0.417 / 0.465 | 0.703 / 0.682 / 0.693 | 0.776 / 0.763 / 0.765 | 932 / - / - |
| X2 | rnd | 1163 / 1110 / 1138 | - / 987 / - | 0.013 / 0.013 / 0.0 | 0.393 / 0.448 / 0.465 | 0.673 / 0.684 / 0.684 | 0.745 / 0.754 / 0.747 | - / - / - |
| Y1 | fresh | 230 / 211 / 232 | 117 / 93 / 108 | 0.087 / 0.087 / 0.075 | 0.458 / 0.463 / 0.427 | 0.861 / 0.873 / 0.85 | 0.994 / 0.998 / 0.983 | 162 / 147 / 157 |
| Y1 | new | 708 / 660 / 677 | 317 / 267 / 286 | 0.025 / 0.013 / 0.037 | 0.165 / 0.138 / 0.117 | 0.584 / 0.572 / 0.568 | 0.959 / 0.949 / 0.959 | 387 / 458 / 472 |
| Y1 | fam | 285 / 244 / 245 | 179 / 116 / 124 | 0.062 / 0.037 / 0.087 | 0.468 / 0.42 / 0.443 | 0.778 / 0.77 / 0.778 | 0.977 / 0.973 / 0.977 | 241 / 188 / 214 |
| Y1 | rnd | 618 / 467 / 537 | 323 / 199 / 285 | 0.0 / 0.062 / 0.0 | 0.19 / 0.22 / 0.282 | 0.649 / 0.671 / 0.679 | 0.961 / 0.975 / 0.972 | 365 / 322 / 337 |
| Y2 | fresh | 126 / 150 / 130 | 60 / 71 / 71 | 0.125 / 0.138 / 0.188 | 0.605 / 0.615 / 0.652 | 0.916 / 0.92 / 0.922 | 0.999 / 0.997 / 0.999 | 124 / 122 / 121 |
| Y2 | new | 308 / 375 / 340 | 135 / 182 / 154 | 0.0 / 0.025 / 0.075 | 0.32 / 0.323 / 0.305 | 0.777 / 0.75 / 0.766 | 0.991 / 0.987 / 0.986 | 195 / 234 / 214 |
| Y2 | fam | 203 / 155 / 178 | 94 / 75 / 90 | 0.013 / 0.037 / 0.075 | 0.487 / 0.565 / 0.53 | 0.855 / 0.876 / 0.866 | 0.997 / 0.993 / 0.995 | 142 / 133 / 144 |
| Y2 | rnd | 271 / 234 / 300 | 123 / 102 / 145 | 0.0 / 0.013 / 0.025 | 0.37 / 0.417 / 0.365 | 0.814 / 0.838 / 0.814 | 0.996 / 0.994 / 0.987 | 179 / 170 / 184 |

Means over all (rule, seed) pairs, and the classification fixed in the plan:

| comparison | pairs | mean err80 a vs b (ratio) | a < b / a > b | mean f100 a vs b | f20 a - b | class |
|---|---|---|---|---|---|---|
| new vs fresh | 12 | 682 vs 224 (3.05) | 0 / 12 | 0.272 vs 0.613 | -0.183 | slower |
| fam vs fresh | 12 | 382 vs 224 (1.71) | 0 / 12 | 0.512 vs 0.613 | -0.156 | slower |
| new vs fam | 12 | 682 vs 382 (1.78) | 0 / 12 | 0.272 vs 0.512 | -0.027 | slower |
| new vs rnd | 12 | 682 vs 503 (1.36) | 0 / 12 | 0.272 vs 0.449 | -0.074 | slower |
| rnd vs fresh | 12 | 503 vs 224 (2.25) | 0 / 12 | 0.449 vs 0.613 | -0.109 | slower |

Same, without X2 (its fresh plateau is 0.75 in the pilot):

| comparison | pairs | mean err80 a vs b (ratio) | a < b / a > b | mean f100 a vs b | f20 a - b | class |
|---|---|---|---|---|---|---|
| new vs fresh | 9 | 410 vs 138 (2.98) | 0 / 9 | 0.305 vs 0.635 | -0.176 | slower |
| fam vs fresh | 9 | 192 vs 138 (1.40) | 0 / 9 | 0.528 vs 0.635 | -0.131 | slower |
| new vs fam | 9 | 410 vs 192 (2.13) | 0 / 9 | 0.305 vs 0.528 | -0.046 | slower |
| new vs rnd | 9 | 410 vs 291 (1.41) | 0 / 9 | 0.305 vs 0.454 | -0.107 | slower |

Per seed, mean err80 over the 4 test rules (new / fam / rnd / fresh):

- seed 1: 673 / 386 / 530 / 222
- seed 2: 680 / 382 / 467 / 217
- seed 3: 693 / 378 / 510 / 232

History curves (err80 per history slot 1-8; f20 in brackets), probes and bits:

- life A seed 1: 146(0.212) 748(0.4) 1722(0.0) 494(0.275) 630(0.237) 893(0.512) 552(0.025) 651(0.487); probe history rules after history 0.72, at end 0.70; end probes Y1 0.675 X1 0.633 Y2 0.1 X2 0.658; bits 519,319 (answer 80,769); 600 s
- life A seed 2: 266(0.225) 486(0.487) 1658(0.013) 800(0.25) 651(0.263) 998(0.487) 544(0.013) 701(0.412); probe history rules after history 0.72, at end 0.75; end probes Y1 0.083 X1 0.717 Y2 0.3 X2 0.75; bits 521,007 (answer 82,350); 628 s
- life A seed 3: 162(0.3) 576(0.412) 1716(0.013) 772(0.25) 566(0.25) 1030(0.512) 542(0.025) 745(0.5); probe history rules after history 0.78, at end 0.77; end probes Y1 0.275 X1 0.475 Y2 0.083 X2 0.808; bits 518,583 (answer 81,267); 610 s
- life B seed 1: 228(0.138) 406(0.4) 155(0.438) 488(0.3) 806(0.15) 247(0.525) 395(0.1) 636(0.263); probe history rules after history 0.83, at end 0.81; end probes X1 0.6 Y1 0.592 X2 0.167 Y2 0.817; bits 520,896 (answer 78,397); 598 s
- life B seed 2: 229(0.163) 390(0.438) 154(0.438) 540(0.263) 793(0.05) 254(0.575) 385(0.087) 768(0.25); probe history rules after history 0.81, at end 0.80; end probes X1 0.55 Y1 0.533 X2 0.3 Y2 0.8; bits 522,120 (answer 78,638); 619 s
- life B seed 3: 271(0.125) 311(0.613) 161(0.512) 529(0.25) 791(0.125) 348(0.5) 420(0.15) 713(0.275); probe history rules after history 0.88, at end 0.83; end probes X1 0.517 Y1 0.433 X2 0.5 Y2 0.85; bits 522,780 (answer 79,499); 607 s
- life RA seed 1: random answers; probe history rules after history nan, at end nan; end probes Y1 0.208 X1 0.525 Y2 0.925 X2 0.833; bits 708,721 (answer 286,932); 578 s
- life RA seed 2: random answers; probe history rules after history nan, at end nan; end probes Y1 0.275 X1 0.483 Y2 0.508 X2 0.775; bits 708,375 (answer 284,225); 596 s
- life RA seed 3: random answers; probe history rules after history nan, at end nan; end probes Y1 0.217 X1 0.433 Y2 0.675 X2 0.625; bits 707,434 (answer 283,782); 589 s
- life RB seed 1: random answers; probe history rules after history nan, at end nan; end probes X1 0.2 Y1 0.225 X2 0.517 Y2 0.833; bits 707,034 (answer 286,161); 574 s
- life RB seed 2: random answers; probe history rules after history nan, at end nan; end probes X1 0.233 Y1 0.308 X2 0.575 Y2 0.708; bits 705,875 (answer 283,257); 591 s
- life RB seed 3: random answers; probe history rules after history nan, at end nan; end probes X1 0.1 Y1 0.317 X2 0.667 Y2 0.667; bits 705,139 (answer 283,023); 593 s

## net12 (seeds [1, 2, 3])

Per test rule and condition: err80 / L80s / f20 / f100 / f500 / last500 (seeds 1, 2, 3; '-' = not reached)

| rule | cond | err80 | L80s | f20 | f100 | f500 | last500 | L80 (100-line) |
|---|---|---|---|---|---|---|---|---|
| X1 | fresh | 526 / 516 / 542 | 247 / 251 / 235 | 0.275 / 0.312 / 0.225 | 0.312 / 0.33 / 0.233 | 0.702 / 0.713 / 0.685 | 1.0 / 1.0 / 0.996 | 300 / 305 / 305 |
| X1 | new | 3250 / 3266 / 3307 | - / - / - | 0.0 / 0.0 / 0.075 | 0.018 / 0.005 / 0.035 | 0.031 / 0.021 / 0.025 | 0.344 / 0.345 / 0.321 | - / - / - |
| X1 | fam | 559 / 522 / 653 | 353 / 358 / 439 | 0.425 / 0.438 / 0.475 | 0.48 / 0.507 / 0.497 | 0.665 / 0.69 / 0.653 | 0.954 / 0.951 / 0.932 | 434 / 433 / 489 |
| X1 | rnd | 1281 / 1315 / 1332 | 577 / 619 / 616 | 0.037 / 0.05 / 0.037 | 0.072 / 0.095 / 0.083 | 0.397 / 0.4 / 0.391 | 0.851 / 0.853 / 0.833 | 695 / 676 / 753 |
| X2 | fresh | 1170 / 1143 / 1203 | 625 / 643 / 662 | 0.075 / 0.125 / 0.113 | 0.11 / 0.14 / 0.09 | 0.475 / 0.494 / 0.473 | 0.828 / 0.831 / 0.828 | 759 / 738 / 791 |
| X2 | new | 3787 / 3805 / 3823 | - / - / - | 0.0 / 0.013 / 0.037 | 0.007 / 0.015 / 0.02 | 0.025 / 0.025 / 0.018 | 0.082 / 0.073 / 0.07 | - / - / - |
| X2 | fam | 1759 / 1764 / 1609 | 853 / 928 / 820 | 0.15 / 0.138 / 0.2 | 0.2 / 0.225 / 0.25 | 0.375 / 0.399 / 0.411 | 0.681 / 0.688 / 0.696 | - / - / - |
| X2 | rnd | 2391 / 2238 / 2270 | - / 889 / 886 | 0.013 / 0.037 / 0.0 | 0.025 / 0.03 / 0.025 | 0.154 / 0.163 / 0.158 | 0.649 / 0.667 / 0.654 | - / - / - |
| Y1 | fresh | 1194 / 1167 / 1139 | 477 / 473 / 447 | 0.05 / 0.062 / 0.05 | 0.072 / 0.083 / 0.043 | 0.392 / 0.406 / 0.408 | 0.933 / 0.931 / 0.933 | 556 / 533 / 521 |
| Y1 | new | 3570 / 3618 / 3621 | - / - / - | 0.013 / 0.013 / 0.037 | 0.018 / 0.022 / 0.045 | 0.029 / 0.022 / 0.033 | 0.185 / 0.169 / 0.157 | - / - / - |
| Y1 | fam | 1349 / 1302 / 1361 | 677 / 646 / 709 | 0.275 / 0.25 / 0.287 | 0.315 / 0.312 / 0.287 | 0.432 / 0.435 / 0.449 | 0.775 / 0.773 / 0.763 | 811 / 905 / 853 |
| Y1 | rnd | 2563 / 2573 / 2566 | 924 / 920 / 929 | 0.0 / 0.062 / 0.0 | 0.007 / 0.018 / 0.01 | 0.091 / 0.094 / 0.091 | 0.6 / 0.583 / 0.597 | 993 / - / - |
| Y2 | fresh | 986 / 924 / 883 | 418 / 404 / 362 | 0.1 / 0.113 / 0.1 | 0.107 / 0.107 / 0.113 | 0.484 / 0.507 / 0.518 | 0.971 / 0.979 / 0.981 | 466 / 462 / 424 |
| Y2 | new | 3674 / 3666 / 3676 | - / - / - | 0.0 / 0.025 / 0.037 | 0.018 / 0.015 / 0.022 | 0.022 / 0.028 / 0.024 | 0.141 / 0.14 / 0.138 | - / - / - |
| Y2 | fam | 1087 / 922 / 1173 | 615 / 565 / 664 | 0.287 / 0.325 / 0.263 | 0.33 / 0.393 / 0.323 | 0.517 / 0.57 / 0.498 | 0.842 / 0.861 / 0.841 | 749 / 620 / 729 |
| Y2 | rnd | 2170 / 2175 / 2158 | 865 / 863 / 847 | 0.0 / 0.013 / 0.037 | 0.043 / 0.033 / 0.043 | 0.178 / 0.169 / 0.176 | 0.688 / 0.695 / 0.691 | 933 / 929 / 921 |

Means over all (rule, seed) pairs, and the classification fixed in the plan:

| comparison | pairs | mean err80 a vs b (ratio) | a < b / a > b | mean f100 a vs b | f20 a - b | class |
|---|---|---|---|---|---|---|
| new vs fresh | 12 | 3589 vs 949 (3.78) | 0 / 12 | 0.020 vs 0.145 | -0.113 | slower |
| fam vs fresh | 12 | 1172 vs 949 (1.23) | 1 / 11 | 0.343 vs 0.145 | +0.159 | no clear difference |
| new vs fam | 12 | 3589 vs 1172 (3.06) | 0 / 12 | 0.020 vs 0.343 | -0.272 | slower |
| new vs rnd | 12 | 3589 vs 2086 (1.72) | 0 / 12 | 0.020 vs 0.040 | -0.003 | slower |
| rnd vs fresh | 12 | 2086 vs 949 (2.20) | 0 / 12 | 0.040 vs 0.145 | -0.110 | slower |

Same, without X2 (its fresh plateau is 0.75 in the pilot):

| comparison | pairs | mean err80 a vs b (ratio) | a < b / a > b | mean f100 a vs b | f20 a - b | class |
|---|---|---|---|---|---|---|
| new vs fresh | 9 | 3516 vs 875 (4.02) | 0 / 9 | 0.022 vs 0.156 | -0.121 | slower |
| fam vs fresh | 9 | 992 vs 875 (1.13) | 1 / 8 | 0.383 vs 0.156 | +0.193 | no clear difference |
| new vs fam | 9 | 3516 vs 992 (3.54) | 0 / 9 | 0.022 vs 0.383 | -0.314 | slower |
| new vs rnd | 9 | 3516 vs 2015 (1.75) | 0 / 9 | 0.022 vs 0.045 | -0.004 | slower |

Per seed, mean err80 over the 4 test rules (new / fam / rnd / fresh):

- seed 1: 3570 / 1188 / 2101 / 969
- seed 2: 3589 / 1128 / 2075 / 938
- seed 3: 3607 / 1199 / 2082 / 942

History curves (err80 per history slot 1-8; f20 in brackets), probes and bits:

- life A seed 1: 837(0.113) 794(0.675) 2535(0.0) 929(0.15) 1529(0.013) 1185(0.5) 1820(0.037) 1042(0.175); probe history rules after history 0.81, at end 0.76; end probes Y1 0.275 X1 0.7 Y2 0.367 X2 0.9; bits 546,354 (answer 123,057); 577 s
- life A seed 2: 817(0.138) 645(0.713) 2493(0.013) 907(0.175) 1569(0.037) 977(0.487) 1884(0.025) 1010(0.212); probe history rules after history 0.80, at end 0.77; end probes Y1 0.225 X1 0.758 Y2 0.358 X2 0.858; bits 546,342 (answer 123,509); 615 s
- life A seed 3: 814(0.15) 715(0.725) 2565(0.013) 954(0.2) 1598(0.062) 1195(0.45) 1905(0.025) 1101(0.212); probe history rules after history 0.82, at end 0.74; end probes Y1 0.258 X1 0.725 Y2 0.325 X2 0.833; bits 545,441 (answer 123,034); 604 s
- life B seed 1: 1082(0.15) 793(0.487) 1734(0.0) 872(0.237) 1427(0.113) 547(0.35) 1871(0.05) 808(0.425); probe history rules after history 0.81, at end 0.77; end probes X1 0.258 Y1 0.575 X2 0.158 Y2 0.925; bits 547,154 (answer 121,917); 598 s
- life B seed 2: 969(0.138) 709(0.512) 1728(0.013) 808(0.35) 1437(0.062) 663(0.375) 1729(0.0) 743(0.425); probe history rules after history 0.83, at end 0.77; end probes X1 0.233 Y1 0.558 X2 0.067 Y2 0.958; bits 547,528 (answer 122,217); 621 s
- life B seed 3: 951(0.113) 812(0.562) 1848(0.013) 973(0.212) 1485(0.05) 595(0.463) 1755(0.013) 737(0.463); probe history rules after history 0.84, at end 0.77; end probes X1 0.317 Y1 0.55 X2 0.117 Y2 0.942; bits 547,412 (answer 122,732); 595 s
- life RA seed 1: random answers; probe history rules after history nan, at end nan; end probes Y1 0.267 X1 0.55 Y2 0.817 X2 0.808; bits 712,756 (answer 306,922); 562 s
- life RA seed 2: random answers; probe history rules after history nan, at end nan; end probes Y1 0.392 X1 0.492 Y2 0.667 X2 0.75; bits 711,498 (answer 305,516); 585 s
- life RA seed 3: random answers; probe history rules after history nan, at end nan; end probes Y1 0.275 X1 0.633 Y2 0.75 X2 0.767; bits 711,818 (answer 305,730); 578 s
- life RB seed 1: random answers; probe history rules after history nan, at end nan; end probes X1 0.242 Y1 0.317 X2 0.567 Y2 0.833; bits 711,763 (answer 306,632); 569 s
- life RB seed 2: random answers; probe history rules after history nan, at end nan; end probes X1 0.408 Y1 0.475 X2 0.592 Y2 0.833; bits 710,725 (answer 304,988); 583 s
- life RB seed 3: random answers; probe history rules after history nan, at end nan; end probes X1 0.333 Y1 0.6 X2 0.65 Y2 0.892; bits 710,946 (answer 305,641); 581 s

## K8 net10 (seeds [1])

Per test rule and condition: err80 / L80s / f20 / f100 / f500 / last500 (seeds 1; '-' = not reached)

| rule | cond | err80 | L80s | f20 | f100 | f500 | last500 | L80 (100-line) |
|---|---|---|---|---|---|---|---|---|
| X1 | fresh | 3618 | - | 0.25 | 0.185 | 0.119 | 0.072 | - |
| X1 | new | 3759 | - | 0.0 | 0.037 | 0.053 | 0.068 | - |
| X1 | fam | 3727 | - | 0.05 | 0.048 | 0.059 | 0.078 | - |
| X1 | rnd | 3881 | - | 0.0 | 0.022 | 0.023 | 0.036 | - |
| X2 | fresh | 3785 | - | 0.062 | 0.052 | 0.044 | 0.063 | - |
| X2 | new | 3695 | - | 0.0 | 0.062 | 0.069 | 0.084 | - |
| X2 | fam | 3654 | - | 0.05 | 0.072 | 0.073 | 0.098 | - |
| X2 | rnd | 3757 | - | 0.037 | 0.043 | 0.053 | 0.067 | - |
| Y1 | fresh | 3525 | - | 0.05 | 0.048 | 0.073 | 0.164 | - |
| Y1 | new | 3591 | - | 0.037 | 0.062 | 0.076 | 0.128 | - |
| Y1 | fam | 3551 | - | 0.013 | 0.072 | 0.089 | 0.135 | - |
| Y1 | rnd | 3796 | - | 0.013 | 0.022 | 0.038 | 0.064 | - |
| Y2 | fresh | 3145 | - | 0.075 | 0.065 | 0.165 | 0.263 | - |
| Y2 | new | 3162 | - | 0.0 | 0.043 | 0.149 | 0.27 | - |
| Y2 | fam | 3146 | - | 0.037 | 0.04 | 0.16 | 0.266 | - |
| Y2 | rnd | 3185 | - | 0.075 | 0.05 | 0.152 | 0.255 | - |

Means over all (rule, seed) pairs, and the classification fixed in the plan:

| comparison | pairs | mean err80 a vs b (ratio) | a < b / a > b | mean f100 a vs b | f20 a - b | class |
|---|---|---|---|---|---|---|
| new vs fresh | 4 | 3552 vs 3518 (1.01) | 1 / 3 | 0.051 vs 0.087 | -0.100 | no clear difference |
| fam vs fresh | 4 | 3520 vs 3518 (1.00) | 1 / 3 | 0.058 vs 0.087 | -0.072 | no clear difference |
| new vs fam | 4 | 3552 vs 3520 (1.01) | 0 / 4 | 0.051 vs 0.058 | -0.028 | no clear difference |
| new vs rnd | 4 | 3552 vs 3655 (0.97) | 4 / 0 | 0.051 vs 0.034 | -0.022 | no clear difference |
| rnd vs fresh | 4 | 3655 vs 3518 (1.04) | 1 / 3 | 0.034 vs 0.087 | -0.078 | no clear difference |

Same, without X2 (its fresh plateau is 0.75 in the pilot):

| comparison | pairs | mean err80 a vs b (ratio) | a < b / a > b | mean f100 a vs b | f20 a - b | class |
|---|---|---|---|---|---|---|
| new vs fresh | 3 | 3504 vs 3429 (1.02) | 0 / 3 | 0.047 vs 0.099 | -0.113 | no clear difference |
| fam vs fresh | 3 | 3475 vs 3429 (1.01) | 0 / 3 | 0.053 vs 0.099 | -0.092 | no clear difference |
| new vs fam | 3 | 3504 vs 3475 (1.01) | 0 / 3 | 0.047 vs 0.053 | -0.021 | no clear difference |
| new vs rnd | 3 | 3504 vs 3621 (0.97) | 3 / 0 | 0.047 vs 0.031 | -0.017 | no clear difference |

Per seed, mean err80 over the 4 test rules (new / fam / rnd / fresh):

- seed 1: 3552 / 3520 / 3655 / 3518

History curves (err80 per history slot 1-8; f20 in brackets), probes and bits:

- life A seed 1: 2971(0.1) 3029(0.037) 3851(0.0) 3673(0.025) 3675(0.037) 2847(0.25) 3819(0.0) 3692(0.025); probe history rules after history 0.11, at end 0.08; end probes Y1 0.092 X1 0.025 Y2 0.275 X2 0.058; bits 708,421 (answer 230,686); 529 s
- life B seed 1: 2255(0.15) 2228(0.062) 3806(0.0) 3636(0.025) 2997(0.075) 2159(0.263) 3707(0.037) 3656(0.025); probe history rules after history 0.23, at end 0.19; end probes X1 0.025 Y1 0.017 X2 0.108 Y2 0.292; bits 714,007 (answer 235,560); 544 s
- life RA seed 1: random answers; probe history rules after history nan, at end nan; end probes Y1 0.0 X1 0.0 Y2 0.267 X2 0.017; bits 812,560 (answer 357,801); 511 s
- life RB seed 1: random answers; probe history rules after history nan, at end nan; end probes X1 0.0 Y1 0.0 X2 0.008 Y2 0.275; bits 811,560 (answer 357,460); 512 s
