Data-scaling part C (t26_calib, M=128, moby/code, T=16k/64k/256k). Run: started 2026-11-24T03:22:31Z, ended 2026-11-24T15:01:43Z (last jsonl line); container resets: 3 seen (03:33Z, 03:45Z, about 05:50Z, after which the batch stayed dead until restarted at 14:50Z); source commit 02fb37f58 (branch claude/newnet-ds-c-sei232). The 14:50Z restart used CKPT_EVERY=90, at the parent session's request (resume.py: saving does not change results).


### moby

| net | T=16k right / bits / nats (table bits) | T=64k right / bits / nats (table bits) | T=256k right / bits / nats (table bits) |
|---|---|---|---|
| M128 | 0.412 / 3.590 / 2.488 (3.298) | 0.454 / 3.103 / 2.151 (2.756) | 0.496 / 2.774 / 1.923 (2.414) |

### code

| net | T=16k right / bits / nats (table bits) | T=64k right / bits / nats (table bits) | T=256k right / bits / nats (table bits) |
|---|---|---|---|
| M128 | 0.431 / 3.817 / 2.646 (3.651) | 0.457 / 3.723 / 2.580 (3.728) | 0.506 / 3.304 / 2.290 (3.634) |

stream lengths seen: [('code', 32006209), ('moby', 1242990)]
