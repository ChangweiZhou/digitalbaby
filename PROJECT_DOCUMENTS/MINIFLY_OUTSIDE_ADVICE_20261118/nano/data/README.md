# The text of the test at the level of nanoGPT

`tinyshakespeare.txt`: the file `input.txt` of the tiny Shakespeare set that nanoGPT's character-level example uses, fetched on 20 November 2026 from
`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` and not changed. 1,115,394 bytes, 65 byte values.

The split is nanoGPT's (`data/shakespeare_char/prepare.py`): the first nine tenths are the stream (1,003,854 bytes, bytes 0 to 1,003,853), the last tenth is never read
(111,540 bytes). Every line of work under `nano/` uses this file and this split and no other preparation of the text.
