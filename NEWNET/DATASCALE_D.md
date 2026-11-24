Run: started 2026-11-24 03:33:20 UTC; ended between 04:19:55 and 05:05:18 UTC (the last result line first appears in the 05:05:18 push; no exact finish time was recorded). Container resets: 1 (about 03:44; restarted at 03:45 and resumed from checkpoints; no later restart was needed). Source commit: a48dd72111a416302d404cdb46b6a2f0585a2547 (branch claude/newnet-ds-d-sei232). Table made with datascale_summary.py after merging claude/magical-albattani-sei232.

### moby

| net | T=16k log-perplexity (nats/byte) / bits / right | T=64k log-perplexity (nats/byte) / bits / right | T=256k log-perplexity (nats/byte) / bits / right |
|---|---|---|---|
| E4 | 2.498 / 3.604 / 0.403 | 2.177 / 3.141 / 0.448 | 1.996 / 2.880 / 0.475 |

### code

| net | T=16k log-perplexity (nats/byte) / bits / right | T=64k log-perplexity (nats/byte) / bits / right | T=256k log-perplexity (nats/byte) / bits / right |
|---|---|---|---|
| E4 | 2.662 / 3.841 / 0.426 | 2.577 / 3.718 / 0.455 | 2.313 / 3.337 / 0.491 |

stream lengths seen: [('code', 32006209), ('moby', 1242990)]
