# E16-WL results

The window net retained operation learning under the every-byte loss. The sign-rule arm reached 92.48–92.98% at stage eight, versus 94.40–95.52% under answer-only loss. Addition and subtraction mod 97 each reached at least 88.5%. P1 missed in all three seeds. Backprop reached 95.98–96.62%, meeting P2 in every seed.

Composition was more sensitive in the sign-rule setup. Held accuracy fell 10.04–24.08 points against its answer-only reference, reaching the predicted 15-point drop in one of three seeds. All three backprop seeds stayed within the predicted loss bound. The two setups use different sampling and Store policies, so their difference cannot isolate the learning rule. The stage losses measure prediction error on prompt and answer positions. They do not measure the relative size or direction of the updates those positions induce.

All six runs passed receipt checks.

Each loss comparison uses its arm’s answer-loss reference. C uses observed replay and a Store. D samples the joint pool of observed examples and has no Store.

| Arm | Seed | Stage 8 mean % | Add97 % | Sub97 % | Final held % | Reference held % | Held loss pp | P1/P2 | P3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 73129 | 92.48 | 88.50 | 89.00 | 72.71 | 82.75 | +10.04 | False | False |
| C | 93683 | 92.98 | 90.17 | 90.00 | 69.33 | 80.96 | +11.62 | False | False |
| C | 116089 | 92.71 | 90.50 | 90.83 | 64.46 | 88.54 | +24.08 | False | True |
| D | 73129 | 96.38 | 94.50 | 92.50 | 89.75 | 89.29 | -0.46 | True | True |
| D | 93683 | 95.98 | 95.83 | 94.83 | 89.00 | 91.25 | +2.25 | True | True |
| D | 116089 | 96.62 | 95.50 | 97.67 | 91.08 | 90.75 | -0.33 | True | True |

The six worker timers sum to 1.720 CPU-hours. Qualification and training took 20.61 minutes after lock acquisition.

Per-stage token-weighted training losses in nats.

| Arm | Seed | Stage | Task | Prompt NLL | Answer NLL | Prompt tokens | Answer tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C | 73129 | 1 | add97 | 1.28565 | 0.06283 | 2752512 | 1179648 |
| C | 73129 | 2 | sub97 | 1.31083 | 0.10018 | 2752512 | 1179648 |
| C | 73129 | 3 | max97 | 1.37578 | 0.08524 | 2752512 | 1179648 |
| C | 73129 | 4 | min97 | 1.42273 | 0.08054 | 2752512 | 1179648 |
| C | 73129 | 5 | add73 | 1.39756 | 0.08655 | 2752512 | 1179648 |
| C | 73129 | 6 | sub73 | 1.41934 | 0.10123 | 2752512 | 1179648 |
| C | 73129 | 7 | max73 | 1.43455 | 0.07697 | 2752512 | 1179648 |
| C | 73129 | 8 | min73 | 1.44901 | 0.07850 | 2752512 | 1179648 |
| C | 73129 | 9 | compose | 1.72582 | 0.36460 | 4456448 | 2359296 |
| C | 93683 | 1 | add97 | 1.28304 | 0.06219 | 2752512 | 1179648 |
| C | 93683 | 2 | sub97 | 1.31748 | 0.10096 | 2752512 | 1179648 |
| C | 93683 | 3 | max97 | 1.37774 | 0.08103 | 2752512 | 1179648 |
| C | 93683 | 4 | min97 | 1.42695 | 0.07644 | 2752512 | 1179648 |
| C | 93683 | 5 | add73 | 1.39824 | 0.08375 | 2752512 | 1179648 |
| C | 93683 | 6 | sub73 | 1.41989 | 0.09368 | 2752512 | 1179648 |
| C | 93683 | 7 | max73 | 1.43296 | 0.07344 | 2752512 | 1179648 |
| C | 93683 | 8 | min73 | 1.44973 | 0.07582 | 2752512 | 1179648 |
| C | 93683 | 9 | compose | 1.72700 | 0.35547 | 4456448 | 2359296 |
| C | 116089 | 1 | add97 | 1.28221 | 0.06284 | 2752512 | 1179648 |
| C | 116089 | 2 | sub97 | 1.31460 | 0.09421 | 2752512 | 1179648 |
| C | 116089 | 3 | max97 | 1.37922 | 0.08325 | 2752512 | 1179648 |
| C | 116089 | 4 | min97 | 1.42781 | 0.07891 | 2752512 | 1179648 |
| C | 116089 | 5 | add73 | 1.39885 | 0.08548 | 2752512 | 1179648 |
| C | 116089 | 6 | sub73 | 1.41958 | 0.09432 | 2752512 | 1179648 |
| C | 116089 | 7 | max73 | 1.43549 | 0.07634 | 2752512 | 1179648 |
| C | 116089 | 8 | min73 | 1.44982 | 0.07882 | 2752512 | 1179648 |
| C | 116089 | 9 | compose | 1.72771 | 0.35821 | 4456448 | 2359296 |
| D | 73129 | 1 | add97 | 1.28567 | 0.05651 | 2752512 | 1179648 |
| D | 73129 | 2 | sub97 | 1.38818 | 0.11858 | 2752512 | 1179648 |
| D | 73129 | 3 | max97 | 1.44863 | 0.08999 | 2752512 | 1179648 |
| D | 73129 | 4 | min97 | 1.49910 | 0.07847 | 2752512 | 1179648 |
| D | 73129 | 5 | add73 | 1.52052 | 0.08279 | 2752512 | 1179648 |
| D | 73129 | 6 | sub73 | 1.54257 | 0.08469 | 2752512 | 1179648 |
| D | 73129 | 7 | max73 | 1.56027 | 0.07225 | 2752512 | 1179648 |
| D | 73129 | 8 | min73 | 1.57688 | 0.07033 | 2752512 | 1179648 |
| D | 73129 | 9 | compose | 1.72518 | 0.26818 | 4644545 | 2547393 |
| D | 93683 | 1 | add97 | 1.28306 | 0.05626 | 2752512 | 1179648 |
| D | 93683 | 2 | sub97 | 1.39184 | 0.11713 | 2752512 | 1179648 |
| D | 93683 | 3 | max97 | 1.45351 | 0.09367 | 2752512 | 1179648 |
| D | 93683 | 4 | min97 | 1.50520 | 0.08535 | 2752512 | 1179648 |
| D | 93683 | 5 | add73 | 1.52356 | 0.08299 | 2752512 | 1179648 |
| D | 93683 | 6 | sub73 | 1.54473 | 0.08702 | 2752512 | 1179648 |
| D | 93683 | 7 | max73 | 1.56243 | 0.07580 | 2752512 | 1179648 |
| D | 93683 | 8 | min73 | 1.57913 | 0.07430 | 2752512 | 1179648 |
| D | 93683 | 9 | compose | 1.71923 | 0.27253 | 4644443 | 2547291 |
| D | 116089 | 1 | add97 | 1.28123 | 0.05726 | 2752512 | 1179648 |
| D | 116089 | 2 | sub97 | 1.38237 | 0.11339 | 2752512 | 1179648 |
| D | 116089 | 3 | max97 | 1.44906 | 0.09117 | 2752512 | 1179648 |
| D | 116089 | 4 | min97 | 1.50114 | 0.08062 | 2752512 | 1179648 |
| D | 116089 | 5 | add73 | 1.52218 | 0.08175 | 2752512 | 1179648 |
| D | 116089 | 6 | sub73 | 1.54321 | 0.08361 | 2752512 | 1179648 |
| D | 116089 | 7 | max73 | 1.56082 | 0.06950 | 2752512 | 1179648 |
| D | 116089 | 8 | min73 | 1.57655 | 0.06871 | 2752512 | 1179648 |
| D | 116089 | 9 | compose | 1.72275 | 0.27800 | 4644563 | 2547411 |
