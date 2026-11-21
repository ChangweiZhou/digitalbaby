# Bytes that are not text: does the one setting hold? (21 November 2026)

**Authorisation and steer.** As `PLAN_1.md` (Mr. Xu's words quoted there), and his words of this night, in Chinese: "大胆假设, 小心求证，继续做实验。" (my translation: "Hypothesise boldly, verify with care, go on experimenting."). The choice of this step is mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written before any of its streams was built or read by any system. Development level; no checker gate, seal or review is claimed.

## 1. Why

Every stream the programme has read is text of some kind: English, source code, a table of numbers written as text, markup. `nano/suite` names this as a limit of its own ("no binary data"). The executed check of that line (roadmap, section 26) found the result less general than six streams suggested. If the system is a general learner of byte streams and not a learner of text, the same program with the same numbers should do something sensible on machine code, fonts, compiled tables and pixels. Nothing in the system was made for them, and nothing is changed for them here.

## 2. The streams (fixed here, before any figure; `kinds6.py` builds them and writes their hashes to `manifests6/`)

Each is 1,500,000 bytes cut from files of this machine; first nine tenths read once, last tenth the test, as in `nano/suite`.

| name | what | recipe |
|---|---|---|
| `elf_c` | machine code and data of a library compiled from C | `/usr/lib/x86_64-linux-gnu/libMagickCore-6.Q16.so.7.0.0`, its first 1,500,000 bytes |
| `elf_go` | machine code compiled from Go | `/usr/bin/docker`, bytes 4,000,000 to 5,500,000 |
| `fonts` | outlines and tables of fonts | the files `*.ttf` in `/usr/share/fonts/truetype/crosextra/` in sorted order, joined, the first 1,500,000 bytes |
| `pyc` | compiled Python (bytecode with its constants) | the files `*.pyc` in `/usr/lib/python3.12/__pycache__/` in sorted order, joined, the first 1,500,000 bytes |
| `magic` | a compiled table of fixed-size records | `/usr/lib/file/magic.mgc`, its first 1,500,000 bytes |
| `pixels` | a picture as raw pixels, row by row, three bytes a pixel | `/opt/google-cloud-sdk/platform/gsutil/third_party/requests/ext/ss.png` decoded, the first 1,500,000 bytes |
| `random` (a control, not a kind) | bytes drawn by chance | `numpy.random.default_rng(20261121).bytes(1500000)` |

## 3. Systems and references

The small system of `PLAN_1.md` (ZA) and the same with the bounded kinds chosen in `PLAN_5.md` (bounds of 32 and 512 on runs of 0 to 3 bytes), each reading a stream alone; both measures. References on the same bytes and the same tenth, computed as `nano/suite` does (the size of the whole compressed less the size of the first nine tenths compressed): xz -9e, bzip2 -9, gzip -9; the best of xz's 27 literal settings (the correction of section 26); and, for the two streams of machine code, xz with its filter for x86 code.

## 4. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** The one setting holds on bytes that are not text: going on, the system with bounded kinds is below xz -9e on the last tenth of at least four of the six kinds, and the bounded kinds pass the rule of `nano/suite` on these six streams too (lower than the small system by 0.001 or more on at least three, none harmed by more than 0.005).

**Refuted if** it is below xz -9e on fewer than three of the six, or the bounded kinds harm a stream by more than 0.005.

| | Statement (going on, second answering cell, last tenth) | Chance |
|---|---|---|
| N1 | The small system is below xz -9e on at least four of six | 0.50 |
| N2 | The system with bounded kinds is below xz -9e on at least four of six | 0.60 |
| N3 | It is below the best of xz's 27 literal settings on at least four of six | 0.50 |
| N4 | The bounded kinds lower the loss by 0.001 or more on at least five of six, and harm none by more than 0.005 | 0.65 |
| N5 | On the picture xz is below both systems (the system has no kind that looks at the row above) | 0.50 |
| N6 | On `elf_c`, xz with its filter for machine code is below the system with bounded kinds | 0.50 |
| N7 | On the bytes drawn by chance both systems cost no more than 0.02 above the 5.545 nats of knowing nothing | 0.55 |
| N8 | With nothing changing, the small system is above xz on at least four of six (as on the streams of text whose end is unlike their beginning) | 0.60 |

## 5. Runs, size, cost, what is not run

Fourteen singles (two systems, seven streams), about 10 to 30 CPU-seconds each on a free machine; the references about 10 CPU-minutes in all. **Not run:** the large system; a life over these streams or over these and the text streams together; any kind made for a kind of data (a row above for pictures, a filter for machine code).

## 6. Outcome

(to be entered after the runs)
