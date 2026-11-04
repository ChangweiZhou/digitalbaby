# MiniFly V202 — bAbI QA1 reproduction

All 15 archived test scores reproduced exactly across five arms and Python hash seeds 1–3. The learned gate scored 84.6–85.7% on exact generated answer plus newline. The random gate scored 3.3–3.9%, predictor alone 21.0–21.3%, all eligible writes 0%, and the task-specific oracle 99.8–100%. The last-mentioned location shortcut scored 52.7%.

The test uses online feedback after each prediction. Earlier development touched the test set. The fixed-seed validation sweep scored 89–90%; the archived validation commands did not specify a hash seed.

Use Python 3 and NumPy, then run `python -B run.py` from this directory. The runner verifies dataset SHA-256 hashes and reruns all test and validation jobs. The recovered learner is included. The original session export is only needed to rerun the optional source recovery script.

## Dataset attribution

The two unmodified text files in data/ are English QA1 train and test from bAbI Tasks v1.2, by Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M. Rush, Bart van Merriënboer, Armand Joulin, and Tomas Mikolov, Facebook Research. See [Towards AI-Complete Question Answering](https://arxiv.org/abs/1502.05698), the [dataset card](https://huggingface.co/datasets/facebook/babi_qa), and the [CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/). The exact copies came from the [same public mirror used in Round 10](https://github.com/anujdutt9/AI-Chatbot/tree/master/bAbI-tasks-v1.2/en). PLAN.json records their hashes.
