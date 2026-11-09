# 9.50 E16-WL

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
