# Genesis V66: Pure R1-Zero Logic

Welcome to **Genesis V66**, a project exploring emergent reasoning in compact neural architectures using "Pure R1-Zero" reinforcement learning. Inspired by the DeepSeek-R1-Zero approach, this agent is trained with **zero-tolerance formatting rewards** and **pure outcome reinforcement** to foster true, emergent reasoning.

## 🚀 Key Features

- **Pure R1-Zero Logic:** The agent receives absolutely $0.00 for the formatting of its thoughts. True reasoning must be emergent, not mathematically bribed.
- **Group Relative Policy Optimization (GRPO):** Trained using a specialized implementation of GRPO to optimize decision-making across diverse logic puzzles.
- **The LiquidBrain Architecture:**
  - **TensorProductHippocampus:** Advanced memory retrieval using tensor-product representations.
  - **Unsupervised Hebbian Plasticity:** Emergent, local learning rules within the neural groups.
  - **TokenMoERouter:** Dynamic routing of tokens to specialized "expert" groups.
  - **Global Workspace Message Passing:** Cross-group communication for coherent reasoning.
- **Trigram Guillotine:** A mathematically airtight Trigram Diversity Ratio to permanently eliminate stuttering or copy-paste "hacks."

## 🧩 The Environment: Storybook Logic

The agent is challenged with "StorybookEnv," a complex logic puzzle where characters move between locations and interact with objects (picking up, dropping). The agent must maintain a mental map of the world and answer "where is" questions by thinking through the sequence of events.

**Example Agent Output:**
> 👩 QUESTION: Where is Harry? Actions: 'say [word]'.
> 👶 Agent: `<think> Harry traveled to the Cupboard. Harry traveled to the Cupboard. Harry is in the Train... wait, no. Harry traveled to Cupboard. </think> say Cupboard`

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/genesis-v66.git
cd genesis-v66
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- PyTorch 2.6+ (with MPS or CUDA support)
- Transformers (for the SmolLM Subword Tokenizer)
- WandB (optional, for logging)

## 📖 Usage

### Training
To start or resume training the agent:
```bash
python genesis_v66.py --batch_size 4 --group_size 4
```

### Configuration
The model behavior can be tuned via `GenesisConfig` in `genesis_v66.py`. Parameters include `d_model`, `n_groups`, `hebb_dim`, and RL-specific settings like `beta_kl` and `clip_eps`.

## 🧠 Technical Introduction

I am [Your Name/Handle], a researcher/developer focused on compact reasoning models. Genesis V66 represents the culmination of several iterations (V59-V65) where I've stripped away "length rewards" and "formatting bribes" to see if a model can learn to "self-correct" its internal thinking purely through outcome success.

The `LiquidBrain` architecture is designed to be highly efficient, with specialized memory and plasticity layers that allow it to adapt to complex state-tracking tasks without the massive parameter counts of standard Transformers.

## 📈 Acknowledgments

- Inspired by **DeepSeek-R1-Zero** and their work on emergent reasoning.
- Uses the **SmolLM** tokenizer for efficient subword encoding.

---
*Note: This repository is intended for research and educational purposes in the field of Reinforcement Learning and Neural Architecture Search.*
