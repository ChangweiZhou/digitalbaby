# MiniFly current research navigation — 2 November 2026

**V180 allocated (DESIGN ONLY).** [V180 phase-blind local plasticity design](minifly_v180_coherence_plasticity_design_20261102/DESIGN.md) | [V180 machine specification](minifly_v180_coherence_plasticity_design_20261102/SPEC.json) | [Evaluation correction](PROJECT_DOCUMENTS/MINIFLY_V180_EVALUATION_CORRECTION_20261102.md) | [Version index](PROJECT_DOCUMENTS/MINIFLY_VERSION_INDEX.json).

**Latest completed numbered experiment: V179 — LIFETIME_DEV_GAPS.** [Full lifetime](minifly_v179_continuous_local_life_dev_20261102/REPORT.md) | [Instrumented train/held divergence](minifly_v179_lifetime_diagnostic_20261102/REPORT.md) | [Paired hidden-plasticity intervention](minifly_v179_plasticity_location_20261102/REPORT.md). V179 showed B training fitting accompanied by worse held loss, while hidden freeze improved later B but impaired continued correction retention. The adopted Full151 core remains unchanged.

V180 tests **one new phase-blind local effective plasticity gate**, against normal learning, fixed lower hidden steps and matched-strength nonselective hidden updates, with the operative learner retaining its parameter/optimizer states across A→B→corrected A. Its new corpus roster and measured retention/EOS contracts are prospective; the 35,000-record budget **requires separate future authorization**. No V180 training, source retrieval or confirmation has been launched. No E2/E3/addition success is claimed.

---

# Repository navigation — 23 October 2026

The consolidated development branch is `codex/minifly-consolidated-20261023`.
The MiniFly [version/results memo](PROJECT_DOCUMENTS/FULL151_VERSION_CENSUS_20261023/MEMO.md)
indexes all 53 reported experiment deliveries, the full 254-row mechanism register,
and the workfolder report census. Historical branches are preserved as tags and
their commits are reachable from the consolidated branch.

The latest [V6-R1 result](minifly_v6_r1_dev_20261023/REPORT.md) is a negative
bounded engineering result, with no demonstrated unseen-addition gain. Its
[publication manifest](minifly_v6_r1_dev_20261023/PUBLICATION_MANIFEST.json)
covers results and code; new raw experiment data are excluded.

The original Genesis V66 introduction below is preserved historical project
documentation. Its task and claims are separate from MiniFly's current addition
experiments.

# DigitalBaby: Emergent Reasoning in Liquid Neural Architectures

**Genesis V66** — A "Pure R1-Zero" experiment in compact reasoning.

DigitalBaby (Genesis V66) is a research project exploring how emergent reasoning (self-correction and logical "chain-of-thought") can arise in specialized neural architectures using **Group Relative Policy Optimization (GRPO)** with absolutely zero formatting or length rewards.

## 🚀 Emergent Intelligence: Level 3 Results

At Level 3 (Storybook Environment), the model begins to show **emergent self-correction**. Because it receives $+5.00$ only for the correct answer and $0.00$ for formatting, the "thought" process is purely functional—the model is *literally thinking to solve the problem*.

### 🌟 Hero Example: The "Wait, no" Moment
In this log, the model realizes it has the wrong location and corrects itself mid-thought:

> **[GRPO] T10893 | 🌟 CORRECT (SELF-CORRECT) 🌟**
> 👩 QUESTION: Where is Harry?
> 👶 **Agent:** `<think> Harry traveled to the Cupboard. Harry traveled to the Cupboard. Harry is in the Train... wait, no. Harry traveled to Cupboard. </think> say Cupboard`

### ⚠️ Honesty & Hallucination
The model also shows "honest" failure modes, where it attempts to correct but fails the logic, leading to negative reinforcement:
> **[GRPO] T10896 | ⚠️ HALLUCINATION**
> 👶 **Agent:** `<think> Draco picked up the Letter. Draco traveled to the Train. Draco is in the Cupboard... wait, no. Draco traveled to Train. </think> say Train`

---

## 🧠 Architecture: The LiquidBrain

Unlike standard Transformers, Genesis V66 uses a "Liquid" architecture designed for state-tracking and long-term reasoning.

```text
[Input Tokens] -> [TokenMoERouter] -> [32 Expert Groups]
                                         |
    +------------------------------------+------------------------------------+
    |                                    |                                    |
[Unsupervised Hebbian]        [Fused Operator Bank]        [Global Message Pass]
  (Local Learning)            (Non-linear Logic)           (Group Communication)
    |                                    |                                    |
    +------------------------------------+------------------------------------+
                                         |
[Memory Tape] <------------> [TensorProductHippocampus] <------------> [Policy Head]
(Long-term Storage)           (Compressed Retrieval)                  (GRPO optimized)
```

### Key Innovations:
- **Pure R1-Zero PRM:** No "bribes" for `<think>` tags. The model only learns to think because thinking leads to the correct answer.
- **Trigram Guillotine:** A mathematical diversity ratio that kills stuttering and "copy-paste" loops by penalizing low-entropy trigram distributions.
- **TensorProductHippocampus:** A specialized memory module that uses tensor-product representations to retrieve character/object states from a compressed "tape."

---

## 🛠️ Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Evaluation (Demo)
To see the model "think" using your trained checkpoint:
```bash
python demo.py --checkpoint genesis_v66_checkpoint.pt
```

### Training
```bash
python genesis_v66.py --batch_size 4 --group_size 4
```

## 📈 Technical Introduction
I am Changwei Zhou, focusing on **Liquid Intelligence**. This repository is an exploration of how we can shrink the "reasoning" capabilities of models like DeepSeek-R1 into compact, efficient architectures that can run on edge devices while maintaining a high logical fidelity.

---
*Note: This is an active research repo. Level 4 training (Multi-object persistence) is currently in progress.*
