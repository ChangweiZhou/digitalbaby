# Full-expression → Full151 → one-byte output, Step 3

**Verdict: `FULL_EXPRESSION_E1_INTERFACE_PASS` under the prespecified four-answer, exact-frame task.** All 16 newly allocated schedule worlds completed, the technical world qualified before science, and the independent receipt audit passed. This is direct teaching and recitation of four visible `a+b=` strings per world. It does **not** establish learned addition, transfer to unseen sums or frames, or multibyte output.

## Question and frozen setup

The prior Step 2 system already allowed the next raw answer byte to route teaching to one of four copied native Full151 value stores. This iteration changed the input from two-letter identifiers to a complete `a+b=` question while retaining FE0, the Full151 checkpoint and plasticity, four output bytes `0`–`3`, the inherited timing, and the byte-controlled teacher router. The model chose a byte at `=` before the answer byte arrived. Each question was directly taught 12 times. Eight worlds used panel A (`0+0=`, `0+1=`, `0+2=`, `1+2=`); eight used panel B (`0+0=`, `0+1=`, `1+1=`, `2+1=`). These fixed panels require some information from the first operand; a last-operand-only rule cannot solve both panels.

The excluded technical world checked KC-address separation for each panel at the answer moment after the initial state and each possible preceding record, as well as exact read/write address equality. The smallest pairwise separation was 42 KCs in panel A and 26 in panel B after a predecessor. An additional descriptive census found no exact wrong-sum KC collisions among 100 single-digit questions at the initial frame; its minimum fixed-second-operand separation was 12 KCs. Address separation is necessary for this assay but does not imply learned performance for 100 facts.

## Behavioral results

The unit for uncertainty is one randomized schedule world, not an individual probe. The table gives end-of-teaching accuracy on the four questions in each world.

| Branch | Target | Correct / 64 | Mean world accuracy | World-level 95% interval |
|---|---|---:|---:|---:|
| W: true answer byte, writes on | True digit | 56 / 64 | 87.5% | 80.6–94.4% |
| N: same bytes, writes off | True digit | 16 / 64 | 25.0% | Zero-variance uncertainty unresolved |
| S: shifted answer byte, writes on | Shifted digit | 56 / 64 | 87.5% | 80.6–94.4% |
| SN: shifted bytes, writes off | Shifted digit | 16 / 64 | 25.0% | Zero-variance uncertainty unresolved |

W beat N and S beat SN in **all 16 worlds**; each strict world-win proportion has an exact two-sided 95% lower bound of 79.4%. S gave **0 / 64** original-answer outputs at the end. The changed answer byte therefore changed what the output compartments learned; sensory exposure to the same bytes without native writing remained at the four-way baseline.

Panel A achieved 32/32 correct in both W and S. Panel B achieved 24/32 in both. In W, the sole panel-B error was `0+1=` in every one of its eight worlds; its output was `2` in three worlds and `3` in five. Other panel-B questions were 8/8 each. This is a repeatable weakness of this exact panel/learning setup, not an exact KC collision: the technical address census separated these cues. The present data do not identify whether overlap, value interference, or another dynamic produced it.

All five prespecified gates passed: W and S world-level lower bounds exceeded 75%; both strict world-win lower bounds exceeded 50%; and both panels' W/S observed accuracies were at least 75%. Panel B sits exactly on that last threshold. Repeating the same four facts under other schedules would only narrow uncertainty about schedule order and would not solve the remaining generalization question.

## Audit and limits

The independent reducer regenerated the fixed panels, record permutations, delivered-byte stream hashes, teacher routes and probe argmax outputs. It checked all 16 world receipts, the source lock, technical qualification, native write counts, clock and read-only probe roster. In-memory tampering of a teacher byte, decoded route, output, cue hash or source digest was rejected. No science failure receipt exists. Source files and prior science worlds were not modified or rerun.

The observed success is **E1 exact-question recitation**: each tested string was taught, the preceding 8 spaces and timing were fixed, output had only four candidate bytes, and `=` identified when to answer. The answer byte in the data stream supplied the teaching signal. Nothing here shows a rule for `1+2` outside its taught frame, changed presentation, autonomously chosen answer timing, a learned stop symbol, or ten-digit/19-sum output. The four stores remain full copied native Full151 learners rather than compact output compartments.

The next scientific gate should use a prospectively fixed, never-taught operand-pair test with both orientations withheld where appropriate, plus no-prior-teaching and exact-cue/frequency/operand baselines. Separately, the systematic `0+1=` panel-B error deserves a small diagnostic on learned value margins and KC overlap before expanding the output alphabet. A larger alphabet or two-byte emitter would add engineering capacity but would not, by itself, establish addition.

Files: [design](DESIGN.md), [runner](output_step3.py), [independent audit](audit.py), [technical receipt](results/TECHNICAL.json), [summary](results/SUMMARY.json), [source lock](results/SOURCE_LOCK.json).
