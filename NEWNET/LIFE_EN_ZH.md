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
