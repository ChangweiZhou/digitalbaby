# Scaling fit (24 Nov 2026)

From scaling_fit.py on datascale_results_{a,b,c,d}.jsonl. N = context cells (16 regions x columns x cells per column); D = bytes read. Seed 1, 12 points per stream.

```

== moby: 12 points (4 nets x 3 data sizes) ==
(1) L = 4.694 -0.0229*log2(N) -0.1325*log2(D)    R2 0.980, rms residual 0.0311
    per doubling: size -0.0229 nats/byte, data -0.1325 nats/byte; data/size ratio 5.8
(2) centred: L = 2.213 -0.0229*n -0.1325*d -0.0061*n*d    R2 0.983, rms 0.0286
    size effect per doubling at D=16k: -0.0108
    size effect per doubling at D=64k: -0.0229
    size effect per doubling at D=256k: -0.0351
(3) power law: L ~ N^-0.0156 * D^-0.0861    R2 0.986
    cC=128 M=32 rc_all=1 vs fit (1), residual by D: 64k -0.036, 16k +0.020, 256k +0.049

== code: 12 points (4 nets x 3 data sizes) ==
(1) L = 4.520 -0.0428*log2(N) -0.0809*log2(D)    R2 0.886, rms residual 0.0511
    per doubling: size -0.0428 nats/byte, data -0.0809 nats/byte; data/size ratio 1.9
(2) centred: L = 2.544 -0.0428*n -0.0809*d -0.0076*n*d    R2 0.896, rms 0.0487
    size effect per doubling at D=16k: -0.0275
    size effect per doubling at D=64k: -0.0428
    size effect per doubling at D=256k: -0.0581
(3) power law: L ~ N^-0.0244 * D^-0.0467    R2 0.877
    cC=128 M=32 rc_all=1 vs fit (1), residual by D: 64k +0.033, 16k -0.043, 256k -0.069
```
