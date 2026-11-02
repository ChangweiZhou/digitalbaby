# MiniFly current research navigation — 2 November 2026

**V181 is the latest allocated experiment (DESIGN ONLY).** [V181 compact learning design](minifly_v181_mini_local_plasticity_design_20261102/DESIGN.md) | [Parameter and resource census](minifly_v181_mini_local_plasticity_design_20261102/PARAMETER_AND_RESOURCE_CENSUS.json) | [Machine-readable spec](minifly_v181_mini_local_plasticity_design_20261102/SPEC.json) | [Scale-correction memo](PROJECT_DOCUMENTS/MINIFLY_V181_MINI_SCALE_CORRECTION_20261102.md) | [Version index](PROJECT_DOCUMENTS/MINIFLY_VERSION_INDEX.json).

**V180 stopped without scientific training: RESOURCE_NOT_ADMITTED.** Its [technical source report](minifly_v180_coherence_plasticity_dev_20261102/REPORT.md) records successful math/implementation checks and the [resource failure](minifly_v180_coherence_plasticity_dev_20261102/RESOURCE_ADMISSION.json). The proposed 4,229,377-parameter model and 16-life campaign were not mini. The original V180 DESIGN/technical source and all V175–V179 negative results are preserved, not retrospectively edited.

**V181 scale correction:** one immutable FE-PERM/5,177-KC sensory birth (without 16 native value banks), sparse KC→H64→T64→257 model with **37,313 actual learned parameters**; 2 worlds ×4 arms under **≤5,000 total training records**, ≤600,000 targets, ≤900 CPU seconds, **≤256 MiB total artifacts**, and a separate **8 MiB single-learner inference-state bound**. All four arms preserve A→B→revised-A continuous weights and optimizer state, correct conditional-revision retention, and report native own EOS. No BPTT, attention, pretrained features or external fitted answerer. V181 has no code implementation or efficacy result; its compact sensory representation and resource claims require measured technical qualification.

**Latest completed numbered efficacy experiment remains V179 — LIFETIME_DEV_GAPS.** [V179 life](minifly_v179_continuous_local_life_dev_20261102/REPORT.md), [instrumented train/held divergence](minifly_v179_lifetime_diagnostic_20261102/REPORT.md), [H/T plasticity package diagnostic](minifly_v179_plasticity_location_20261102/REPORT.md). Adopted Full151 core unchanged. No V182 is allocated.

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
