# MiniFly V205 — Round 12 QA1 and QA2 reproduction

All 54 final-roster jobs completed with Python hash seed 4. The recovered learners match the source hashes in the Round 12 memo.

Frozen QA1 reached 100% on reversed wording, 99.8% on new locations, and 99.5% on the changed-prefix stories without gold answers. On qfound, frozen performance was 52.7% and adaptive question-frequency scoring reached 99.2%.

QA2 reached 100% on the original test and 95.8% with new person names, 88.3% with new object names, and 87.7% with both changed. The fixed channel control was 85.2%. Both curriculum orders scored 100% on both original test sets. The renamed-object results were lower than the original three-seed results. V205 varies the hash seed while retaining the benchmark and transformations.

## Reproduction

Use Python 3 and NumPy. From the repository root:

```sh
ln -s minifly_v202_babi_qa1 minifly-v202
ln -s minifly_qa1_frozen_generalization minifly-qa-generalization
cd minifly_v205_round12_qa1_qa2
python -B run.py
```

The runner regenerates the derived evaluation files and verifies the original dataset hashes. The optional recover.py requires the original Round 12 session export. The recovered sources are included for reruns.

## Dataset attribution

The two unmodified files in data/ are English QA2 train and test from bAbI Tasks v1.2, by Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M. Rush, Bart van Merriënboer, Armand Joulin, and Tomas Mikolov, Facebook Research. See the [dataset card](https://huggingface.co/datasets/facebook/babi_qa), the [CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/), and the [Round 12 mirror](https://github.com/anujdutt9/AI-Chatbot/tree/master/bAbI-tasks-v1.2/en). QA1 files and attribution are in V202.
