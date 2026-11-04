# MiniFly V203 — bAbI QA1 Binary Write Gate

All 20 Round 11 test and validation runs reproduced the archived scores. Frozen gate-scaled writes scored 71.3–74.1%. Frozen full-strength writes scored 99.8–100%, and online full-strength writes scored 99.4–99.6%. Frozen validation improved from 90.5% to 99.0%.

Both frozen variants selected all 2,000 location-start positions and none of the 58,104 other eligible positions. The audit verified unchanged predictor, gate and answer-memory hashes across frozen evaluation. The episodic store continued to write statements. The binary intervention was prompted by inspection of the same QA1 test stream, so these results are a reproduction on the existing task.

## Reproduction

Use Python 3 and NumPy. From the repository root, create the local alias used by the source-locked runner and then run the bundle:

```sh
ln -s minifly_v202_babi_qa1 minifly-v202
cd minifly_v203_babi_qa1_binary_gate
python -B run.py
```

The runner verifies both QA1 data hashes against PLAN.json. V202 contains the original data and attribution. The optional recovery script requires the original Round 11 session export. The recovered learners are already included for reruns. RESULTS.json contains every scored row and frozen-state audit.
