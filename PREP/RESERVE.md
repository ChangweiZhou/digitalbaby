# The reserve of never-taught instances, divided beforehand (`PREP/reserve.py`)

**What it is.** This is preparation for steps 5 and 6a of the roadmap (`ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md`, section 5, condition 4, and section 6). Condition 4 says that never-taught instances are set aside before the stream is made, that each is asked once in a life, and that the reserve is divided beforehand twice: by when an instance is asked, and by which of its parts were taught in other instances. The program makes that division for the first family and computes what two countermodels score on every part of it. It trains nothing and reads no network. It uses the standard library and numpy, and runs in about two seconds:

    python3 PREP/reserve.py --check       # prints the tables, writes reserve_<seed>.json; --check asserts 3,960 taught and 3,961 reserve pairs a task

**The reserve.** For each of the four tasks of the first family the record splits the 7,921 ordered pairs of numbers below 89 in two halves: `split(task, .5, seed)` of `check_step1_brief.py`, which is the record's `exp11_cell.split`. The first half, 3,960 pairs, is the taught half that the stream of step 1 shows. The second half, 3,961 pairs, never appears in the stream and is the reserve. The tasks are:
- `v` max and `w` min, whose answer is one of the two numbers (step 5);
- `p` addition and `q` subtraction modulo 89 (step 6a).

**The groups.** Each reserve pair (a, b) of a task carries three marks, each taken against the taught pairs of the same task.

| Mark | Holds when |
|---|---|
| `T` | the tens digits of a and b occur together in some taught pair (a′, b′) |
| `U` | the units digits of a and b occur together in some taught pair |
| `R` | the reversed pair (b, a) is taught |

The group is the set of marks that hold, for example `T U R` or `T U -`.

**The parts.** Within each task and group the pairs are shuffled by `Random(f"reserve|{seed}|{op}|{group}")` and dealt in turn to three parts:
- `E`, asked early;
- `D`, asked after a delay;
- `O`, asked after other learning.

Every group is therefore spread over the three parts as evenly as its count allows: 1,320 or 1,321 pairs a part for each task. The division is written to `reserve_<seed>.json`, with the part of every pair and the group of every pair, for use by later briefs.

**The countermodels.** Both know only the taught pairs of the task.
- *The table* answers a reserve pair from the table of taught pairs and their answers. Its key is (a, b), or the unordered {a, b} where the operation is symmetric (addition, max, min), so a pair whose reverse was taught is answered from the table. A pair whose key is absent is answered by the frequencies.
- *The frequencies* answer the most frequent tens digit and the most frequent units digit among the taught answers of the task, place by place.

The score is the share of pairs whose two-digit answer is exactly right. The share of right tens digits and of right units digits is given beside it, and for addition and subtraction also the score on the pairs that need the modulus.

**What the output shows** (`reserve_output.txt`, the same for the three seeds up to a few pairs):

1. **The marks `T` and `U` hold for every reserve pair of every task and seed.** With half the pairs taught, every one of the 81 pairs of tens digits occurs in at least 33 taught pairs of its task, and every one of the 100 pairs of units digits in at least 24. At this split the grouping by parts therefore reduces to one division: **`R` (the reverse is taught) against not-`R`**.
   - About half the reserve is `R`: 1,883 to 2,025 pairs a task.
   - The pairs with a = b (40 to 52 a task) are never `R`, because their reverse is themselves.
   - A finer division, needed if step 5 is to test new combinations of parts and not merely reversals, must use a stricter part than "occurs somewhere". Examples: a tens pair taught fewer than k times, or a split with fewer pairs taught. This is a question for the brief of step 5, not decided here.
2. **The table scores 100% on every `R` pair of addition, max and min**, and therefore 48 to 52% on their whole reserve. On subtraction it scores no better than the frequencies (0.5 to 1.4%), because subtraction is not symmetric. A tissue that answers max, min or addition on `R` pairs has shown nothing beyond a symmetric table. Only the not-`R` pairs, and every pair of subtraction, can carry evidence of generalisation, and on these the table falls back to the frequencies.
3. **The frequencies score 0.2 to 3.4% exactly** on every group and part (0.7 to 3.1% on a whole task). They get 7 to 25% of tens digits and 7 to 14% of units digits right; the higher tens figures are for max and min, whose answers lean to large and small numbers respectively. On the pairs of addition and subtraction that need the modulus, the frequencies score 0.2 to 1.7%.
4. **The three parts are alike** for every task, group and seed: the table's score differs between `E`, `D` and `O` by at most 0.7 points on a whole task. A difference between the times of asking in a tissue is therefore not a difference between the parts' contents.

**What it does not do.**
- It does not render the reserve pairs as items (the letters and their number are the stream's business, and step 6b varies them).
- It does not fix when in a life `E`, `D` and `O` are asked.
- It does not add the countermodel "one of the two numbers" for max and min (which scores 50% by guessing), nor the comparator of condition 5. All three are for the brief of step 5.

**Files:** `PREP/reserve.py`, `PREP/reserve_output.txt`, and `PREP/reserve_73129.json`, `reserve_93683.json` and `reserve_116089.json` (the division), with this description.
