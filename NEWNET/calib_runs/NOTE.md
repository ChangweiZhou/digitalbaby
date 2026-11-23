The first four lines of res.jsonl (23 Nov, seed 1, K8, T=16000) were written before the script recorded eta_s/eta_t; the three moby
lines are the runs (eta_s, eta_t) = (1.0, 0.01), (3.0, 0.003), (0.3, 0.03) in an order not recorded; the code line is (1.0, 0.01).
Bits per byte (nats = bits x 0.693): clip-normalised outputs moby 3.787 / code 4.481; learned temperature best moby 3.589 (2.49 nats),
code 3.947 (2.74 nats); softmax readout over active cells 3.88-10.1 (worse; underfits or diverges at 16 kB); count table moby 3.298
(2.29 nats), code 3.651 (2.53 nats).
