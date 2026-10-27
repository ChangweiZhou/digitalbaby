# MiniFly mechanistic synthesis — V159–V168

**Date:** 2026-10-26 (discussion consolidation)  
**Type:** Research memos; documentation only.  
**Source snapshots:** Claude branch `7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c`; Codex branch `bb7c80c0cfa0c08972648d3f74bb86c7e2e4aa8d`.  
**Scientific status:** No experiment launched by these memos, no architecture adopted, no frozen protocol overwritten. References and summary values come from published development reports and committed code, not an independent rerun.

This set consolidates the last rounds of mathematical/architecture discussion. Read together with [the earlier post-V156 roadmap](../MINIFLY_RESEARCH_ROADMAP_20261026/MEMO.md).

| Memo | Purpose |
|---|---|
| [01 — Evidence and architecture](01_EVIDENCE_AND_ARCHITECTURE.md) | What V159–V168 actually proved, bounded controls and what remains unproved |
| [02 — Mathematical puzzle](02_MATHEMATICAL_DIAGNOSIS.md) | Representability, functional state, conditional prediction, memory composition and temporal credit |
| [03 — Investigation plan](03_DISCRIMINATING_EXPERIMENTS.md) | Concrete experiments, falsifiers, priorities and ownership |

**Main synthesis:** Locally broadcast task error can create useful transferred features on the tested text pair (V165/V167). An explicit memory-to-working-state path permits two-hop retrieval under BPTT (V166). But approximate online temporal credit on that same path (V168) has not reached the BPTT capability. These are separate achievements and gaps. They do not establish native Full151 integration, human-level learning, autonomous byte/EOS addition, or E3 transfer.

**Terminology:** BP-H is a gradient-trained reference; DFA-H has a BP-trained top; V167 LOCAL trains its source network without cross-layer backprop, but its cross-text evaluation probe is BP-trained. V166/V168 use structured synthetic episodes with provided study/query/answer markers, not raw-byte addition.

Source anchors:
- [V159 report](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v159_pfeed_dev_v1_20261026/REPORT.md)
- [V162 report](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v162_text_dev_v1_20261026/REPORT.md)
- [V163 report](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v163_sslcola_dev_v1_20261026/REPORT.md)
- [V164 report](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v164_cola_dev_v1_20261026/REPORT.md)
- [V165 report](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v165_dfa_dev_v1_20261026/REPORT.md)
- [V166 report](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v166_memloop_dev_v1_20261026/REPORT.md)
- [V167 report](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v167_local_dev_v1_20261026/REPORT.md)
- [V168 report](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v168_kfloop_dev_v1_20261026/REPORT.md)
