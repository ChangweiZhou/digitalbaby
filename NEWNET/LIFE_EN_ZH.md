# English and Chinese in one life (22 November 2026; seed 1)

Owner, this session: "换成两个完全不同的文本，例如英文和中文的文本。……网络本身对超参数的学习暂时推迟。" The two best structures of the
architecture search (`SEARCH_SUMMARY.md`), unchanged except 256 byte values: K8 (two pathways) and N1 (chain with shortcuts).
English: tinyshakespeare. Chinese: Hong Lou Meng, UTF-8 (three bytes a character), from a Gutenberg mirror
(gutenberg.pglaf.org/2/4/2/6/24264/24264-0.txt; not tracked). Life: 16,000 bytes of the first text, 16,000 of the second, 2,000 of the
first again (`t13_life8.py life=enzh` / `zhen`). Tests with learning off on 10,000 bytes of each text never read. Control: the second
text only (its score on the first text is 0.000-0.004, so what remains of the first text is kept, not carried over).

| life | structure | first text after it | after the second (kept) | second text after it | second text alone | first text after the return |
|---|---|---|---|---|---|---|
| English -> Chinese | K8 | 0.371 | 0.364 (98%) | 0.337 | 0.347 | 0.362 |
| English -> Chinese | N1 | 0.381 | 0.326 (86%) | 0.334 | 0.331 | 0.369 |
| Chinese -> English | K8 | 0.336 | 0.311 (93%) | 0.367 | 0.370 | 0.350 |
| Chinese -> English | N1 | 0.335 | 0.255 (76%) | 0.356 | 0.368 | 0.344 |

Readings: (1) between texts that share almost no bytes, little is lost (K8 98% and 93%), far less than between two English texts; (2)
K8 keeps much better than N1 in both orders, the first difference found between the two leaders; (3) the first text neither helps nor
hinders the second (-0.012 to +0.003 against the second alone); (4) 2,000 bytes of the first text bring it back to or above its level.
Seeds 2 and 3 are running.

## Seeds 2 and 3

| kept share of the first text through the second | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| English -> Chinese, K8 | 0.98 | 0.96 | 0.99 | **0.98** |
| English -> Chinese, N1 | 0.86 | 0.88 | 0.88 | 0.87 |
| Chinese -> English, K8 | 0.93 | 0.91 | 0.91 | **0.91** |
| Chinese -> English, N1 | 0.76 | 0.79 | 0.81 | 0.79 |

(Seeds 2 / 3: first text after it and after the second: English->Chinese K8 0.371 -> 0.358, 0.359 -> 0.356; N1 0.368 -> 0.325, 0.363 ->
0.321; Chinese->English K8 0.341 -> 0.311, 0.337 -> 0.305; N1 0.344 -> 0.272, 0.335 -> 0.271. The second text after it: K8 0.333 /
0.331 and 0.360 / 0.348, N1 0.345 / 0.335 and 0.367 / 0.355. After the return: within 0.01 of the first text's level, or above.)
K8 keeps better than N1 on every seed and in both orders: confirmed.

## Where the first text is lost (`t19_putback.py`, seed 1): put each part back as it was after the first text

| life | structure | first text after it / after the second | context cells' connections back | context outputs back | holders' outputs back |
|---|---|---|---|---|---|
| Chinese -> English | K8 | 0.336 / 0.313 | 0.294 | 0.320 | **0.337** |
| Chinese -> English | N1 | 0.335 / 0.266 | 0.247 | 0.274 | **0.324** |
| English -> Chinese | K8 | 0.371 / 0.365 | 0.367 | 0.364 | 0.364 |
| English -> Chinese | N1 | 0.381 / 0.339 | 0.337 | 0.338 | **0.373** |

The loss is in the holders' output strengths: put back, the first text returns almost fully. Recycling of context cells is not the
cause (putting the context cells' connections back does not help). Some of the holders' expansion cells fire in both languages, and
their reading is rewritten, as between two English texts. Next: a sparser and wider holders' expansion, where the two languages'
holder states should differ much more than two English texts do.

**A sparser, wider holders' expansion** (`h_g_k=0.005 h_g_fan=16`; seed 1): N1 Chinese -> English 0.327 -> 0.286 (kept 87%; before 76%);
N1 English -> Chinese 0.377 -> 0.355 (94%; before 86%); N1 with 0.2% firing, Chinese -> English 0.325 -> 0.287 (88%); K8 Chinese -> English
0.335 -> 0.315 (94%; before 93%). Between two very different languages the sparser holders' expansion separates them and keeps N1's first
language much better (+8 to +11 points), at a small cost in learning it (0.327 against 0.335); K8 was already at 93% and hardly changes.
What is left of the loss is then shared between the context outputs and the holders' outputs. (Between two English texts and within
letters, the same change did nothing: there the holder states differ too little.)
