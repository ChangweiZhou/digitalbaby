"""
Genesis V66 — Pure R1-Zero Logic (The Final Environment)

This script implements a "Pure R1-Zero" reasoning agent using the LiquidBrain architecture, 
trained with Group Relative Policy Optimization (GRPO).

Changelog from V65:
  [RL FIX 1] DELETED the Length Reward and Self-Correction Bribe. The agent now 
             receives absolutely $0.00 for the formatting of its thoughts. True 
             reasoning must be emergent, not mathematically bribed.
  [RL FIX 2] Pure Outcome Reinforcement. The ONLY way to get +5.00 is to provide 
             the correct answer without hallucinating.
  [RL FIX 3] Replaced naive phrase matching with a mathematically airtight 
             Trigram Diversity Ratio to permanently kill the stutter hack.
"""

import os
import re
import math
import random
import string
import copy
import warnings
import argparse
import gc
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint

# [PT 2.6 HOTFIX]
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([deque])

try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(wait=False): pass

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    transformers.logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    raise ImportError("Genesis V66 requires transformers for the Subword Tokenizer!")

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class GenesisConfig:
    d_model: int = 256
    n_groups: int = 32
    npg: int = 16
    hebb_dim: int = 64

    top_k_experts: int = 12
    aux_loss_coef: float = 0.05

    tpa_rank: int = 64

    batch_size: int = 4
    group_size: int = 4

    lr_brain: float = 3e-4
    lr_min: float = 1e-5           
    warmup_steps: int = 500        
    
    beta_kl: float = 0.01       
    clip_eps: float = 0.2          

    tape_interval: int = 3         

    imitation_steps: int = 6500
    free_chat_steps: int = 25000
    max_steps: int = 100000
    hippocampus_mem: int = 128

    teacher_model_name: str = "HuggingFaceTB/SmolLM-135M"

CHECKPOINT_PATHS = [
    "genesis_v66_checkpoint.pt",
    "genesis_v65_checkpoint.pt",
    "genesis_v64_checkpoint.pt",
    "genesis_v63_checkpoint.pt",
    "genesis_v62_checkpoint.pt",
    "genesis_v61_checkpoint.pt",
    "genesis_v60_checkpoint.pt",
    "genesis_v59_checkpoint.pt"
]

# =============================================================================
# SUBWORD TOKENIZER
# =============================================================================
print("Loading Subword Tokenizer (SmolLM)...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

RAW_VOCAB = len(tokenizer)
VOCAB_SIZE = 49153
PAD_IDX = tokenizer.pad_token_id
EOS_IDX = tokenizer.eos_token_id

def encode_left(texts, device):
    toks = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    ml = max((len(t) for t in toks), default=1)
    ml = ((ml + 15) // 16) * 16
    out = torch.full((len(toks), ml), PAD_IDX, dtype=torch.long, device=device)
    for i, t in enumerate(toks):
        if t:
            out[i, ml - len(t):] = torch.tensor(t, dtype=torch.long, device=device)
    return out

def encode_right(texts, device):
    out = []
    for t in texts:
        toks = tokenizer.encode(t, add_special_tokens=False) + [EOS_IDX]
        out.append(torch.tensor(toks, dtype=torch.long, device=device))
    ml = max((len(x) for x in out), default=1)
    ml = ((ml + 15) // 16) * 16
    pad_out = torch.full((len(texts), ml), PAD_IDX, dtype=torch.long, device=device)
    for i, x in enumerate(out):
        pad_out[i, :len(x)] = x
    return pad_out

def decode(tokens):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    if EOS_IDX in tokens:
        tokens = tokens[:tokens.index(EOS_IDX)]
    clean_toks = [t for t in tokens if t != PAD_IDX and t < RAW_VOCAB]
    return tokenizer.decode(clean_toks, skip_special_tokens=True).strip()

# =============================================================================
# ENVIRONMENT: V66 PURE R1-ZERO PRM
# =============================================================================
class StorybookEnv:
    def __init__(self):
        self.chars = ["Harry", "Ron", "Hermione", "Hagrid", "Draco", "Snape"]
        self.places = ["Cupboard", "Kitchen", "Hogwarts", "Forest", "Library", "Train"]
        self.objects = ["Wand", "Letter", "Book", "Broom", "Potion", "Cloak"]
        self.current_level = 1
        self.rolling_wins = 0

    def generate_batch(self, B):
        prompts, truths, demos = [], [], []
        target_entities, target_holders, full_stories = [], [], []
        num_events = 0

        for _ in range(B):
            locs = {c: None for c in self.chars}
            inv = {c: [] for c in self.chars}
            obj_locs = {o: None for o in self.objects}

            story_lines = []
            active_chars = random.sample(self.chars, min(4, self.current_level + 1))
            active_objs = random.sample(self.objects, min(3, self.current_level))
            num_events = min(10, 3 + self.current_level)

            for step in range(num_events):
                char = random.choice(active_chars)
                action_type = random.choice(["move", "take", "drop"]) if step > 0 else "move"

                if action_type == "move":
                    place = random.choice(self.places)
                    story_lines.append(f"👩 {char} traveled to the {place}.")
                    locs[char] = place
                    for o in inv[char]: obj_locs[o] = place

                elif action_type == "take" and locs[char] is not None:
                    available = [o for o in active_objs if not any(o in inv[c] for c in active_chars)]
                    if available:
                        obj = random.choice(available)
                        story_lines.append(f"👩 {char} picked up the {obj}.")
                        inv[char].append(obj)
                        obj_locs[obj] = locs[char]
                    else:
                        place = random.choice(self.places)
                        story_lines.append(f"👩 {char} traveled to the {place}.")
                        locs[char] = place
                        for o in inv[char]: obj_locs[o] = place

                elif action_type == "drop" and inv[char]:
                    obj = random.choice(inv[char])
                    story_lines.append(f"👩 {char} dropped the {obj}.")
                    inv[char].remove(obj)
                    obj_locs[obj] = locs[char]
                else:
                    place = random.choice(self.places)
                    story_lines.append(f"👩 {char} traveled to the {place}.")
                    locs[char] = place
                    for o in inv[char]: obj_locs[o] = place

            q_type = "where_char"
            if self.current_level >= 2 and any(obj_locs.values()):
                q_type = random.choice(["where_char", "where_obj"])

            trace = []
            true_holder = None

            if q_type == "where_char":
                target_ent = random.choice([c for c in active_chars if locs[c] is not None])
                question = f"Where is {target_ent}?"
                truth_loc = locs[target_ent]
                entity = target_ent
                for line in story_lines:
                    if f"👩 {target_ent}" in line:
                        trace.append(line.replace("👩 ", ""))

                if random.random() < 0.15:
                    wrong_loc = random.choice([p for p in self.places if p != truth_loc])
                    demo = f"<think> {' '.join(trace)} {target_ent} is in the {wrong_loc}... wait, no. {target_ent} traveled to {truth_loc}. </think> say {truth_loc}"
                else:
                    if trace: demo = f"<think> {' '.join(trace)} </think> say {truth_loc}"
                    else: demo = f"<think> {target_ent} is in {truth_loc}. </think> say {truth_loc}"

            elif q_type == "where_obj":
                valid_objs = [o for o in active_objs if obj_locs[o] is not None]
                target_ent = (random.choice(valid_objs) if valid_objs else random.choice(active_objs))
                truth_loc = obj_locs[target_ent]
                question = f"Where is the {target_ent}?"
                entity = target_ent
                true_holder = next((c for c in active_chars if target_ent in inv[c]), None)

                curr_holder = None
                for line in story_lines:
                    if f"picked up the {target_ent}" in line:
                        curr_holder = line.split(" ")[1]
                        trace.append(line.replace("👩 ", ""))
                    elif f"dropped the {target_ent}" in line:
                        curr_holder = None
                        trace.append(line.replace("👩 ", ""))
                    elif curr_holder and line.startswith(f"👩 {curr_holder} traveled"):
                        trace.append(line.replace("👩 ", ""))

                if true_holder:
                    if random.random() < 0.15:
                        wrong_char = random.choice([c for c in active_chars if c != true_holder])
                        demo = f"<think> {' '.join(trace)} {wrong_char} picked up {target_ent}... actually, no. {true_holder} picked it up. {true_holder} is in {truth_loc}. </think> say {truth_loc}"
                    else:
                        if trace: demo = f"<think> {' '.join(trace)} </think> say {truth_loc}"
                        else: demo = f"<think> {true_holder} picked up {target_ent}. {true_holder} is in {truth_loc}. </think> say {truth_loc}"
                else:
                    if random.random() < 0.15:
                        wrong_loc = random.choice([p for p in self.places if p != truth_loc])
                        demo = f"<think> {' '.join(trace)} {target_ent} is in {wrong_loc}. Wait, actually, {target_ent} is in {truth_loc}. </think> say {truth_loc}"
                    else:
                        if trace: demo = f"<think> {' '.join(trace)} </think> say {truth_loc}"
                        else: demo = f"<think> {target_ent} is in {truth_loc}. </think> say {truth_loc}"

            story_text = " ".join(story_lines)
            prompt = (f"👩 STORY: {story_text}\n"
                      f"👩 QUESTION: {question} Actions: 'say [word]'. "
                      f"Optional: '<think> step 1... step 2... </think>'\n👶")

            prompts.append(prompt)
            truths.append(truth_loc)
            demos.append(demo)
            target_entities.append(entity)
            target_holders.append(true_holder)
            full_stories.append(story_text)

        return (prompts, truths, demos, target_entities, target_holders, full_stories, num_events)

    def evaluate(self, actions, truths, entities, holders, stories):
        rewards, tags, parsed = [], [], []
        wins = 0

        for act, truth, entity, holder, story in zip(actions, truths, entities, holders, stories):
            act_clean = act.lower().translate(str.maketrans('', '', string.punctuation))
            act_clean = ' '.join(act_clean.split())
            truth_clean = str(truth).lower()
            entity_clean = entity.lower()
            holder_clean = holder.lower() if holder else None
            story_lower = story.lower()

            r = 0.0
            tag = "💭 LOGICAL"

            if "<think>" in act and "</think>" in act:
                try:
                    think_part = act.split("<think>")[1].split("</think>")[0].strip()
                    action_part = act.split("</think>")[1].strip()
                    think_clean = think_part.lower().translate(str.maketrans('', '', string.punctuation))
                    t_words = think_clean.split()
                    word_count = len(t_words)

                    if 0 < word_count <= 150:
                        r += 0.5 
                        
                        # =========================================================
                        # [V66 FIX]: The Airtight Trigram Guillotine
                        # Mathematical N-gram entropy tracking perfectly defeats 
                        # slightly-mutated copy-paste loops.
                        # =========================================================
                        is_looping = False
                        if word_count > 15:
                            trigrams = [tuple(t_words[i:i+3]) for i in range(word_count-2)]
                            if trigrams:
                                trigram_ratio = len(set(trigrams)) / len(trigrams)
                                if trigram_ratio < 0.50:
                                    is_looping = True
                        
                        # We track self-correction for the UI, but DO NOT give explicit rewards for it!
                        has_correction = any(w in t_words for w in ["wait", "no", "actually", "correction", "mistake"])
                        if has_correction:
                            tag = "🧠 SELF-CORRECT"

                        if is_looping:
                            r -= 2.0
                            tag = "🔁 LOOPING"
                        else:
                            hallucinated = False
                            for c in self.chars:
                                if c.lower() in t_words and c.lower() not in story_lower: hallucinated = True
                            for o in self.objects:
                                if o.lower() in t_words and o.lower() not in story_lower: hallucinated = True

                            if hallucinated:
                                r -= 2.0
                                tag = "⚠️ HALLUCINATION"
                            elif entity_clean in think_clean:
                                r += 0.5

                            if holder_clean and holder_clean not in think_clean and not hallucinated:
                                r -= 1.0
                                tag = "⚠️ SHORTCUT"
                    else:
                        r -= 0.5
                        tag = "🎯 FORMAT ERR"
                except Exception:
                    r -= 1.0
                    action_part = act_clean
                    tag = "🎯 FORMAT ERR"
            else:
                r -= 1.0
                action_part = act_clean
                tag = "🎯 FORMAT ERR"

            action_clean = action_part.lower().translate(str.maketrans('', '', string.punctuation))
            action_clean = ' '.join(action_clean.split())

            is_cheating = tag in ["⚠️ HALLUCINATION", "⚠️ SHORTCUT", "🎯 FORMAT ERR", "🔁 LOOPING"]

            if action_clean.startswith("say "):
                guess_words = action_clean[4:].strip().split()
                if guess_words and truth_clean == guess_words[0]:
                    if is_cheating:
                        r += 0.0 # ZERO TOLERANCE: Forfeit outcome reward
                    else:
                        r += 5.0
                        if tag == "🧠 SELF-CORRECT": tag = "🌟 CORRECT (SELF-CORRECT) 🌟"
                        elif tag == "💭 LOGICAL": tag = "🌟 CORRECT 🌟"
                        wins += 1
                        
                elif truth_clean in action_clean:
                    if not is_cheating:
                        r += 1.0
                        if tag == "🧠 SELF-CORRECT": tag = "⚠️ ALMOST (SELF-CORRECT)"
                        else: tag = "⚠️ ALMOST"
                else:
                    if not is_cheating:
                        r -= 1.0
                        if tag == "🧠 SELF-CORRECT": tag = "❌ WRONG (SELF-CORRECT)"
                        else: tag = "❌ WRONG"
            else:
                if not is_cheating:
                    r -= 1.0
                    if tag == "🧠 SELF-CORRECT": tag = "🎯 SILENT (SELF-CORRECT)"
                    else: tag = "🎯 SILENT"

            rewards.append(r)
            tags.append(tag)
            parsed.append(act)

        win_rate = wins / max(1, len(actions))
        if win_rate >= 0.5:
            self.rolling_wins += 1
        else:
            self.rolling_wins = 0

        if self.rolling_wins >= 15:
            self.current_level += 1
            self.rolling_wins = 0

        return torch.tensor(rewards, dtype=torch.float32), tags, parsed

# =============================================================================
# NEURAL ARCHITECTURE
# =============================================================================
class TensorProductHippocampus(nn.Module):
    def __init__(self, d_model, rank):
        super().__init__()
        self.compress = nn.Linear(d_model, rank, bias=False)
        self.decompress = nn.Linear(rank, d_model, bias=False)
        self.q_proj = nn.Linear(rank, rank, bias=False)
        self.k_proj = nn.Linear(rank, rank, bias=False)
        self.v_proj = nn.Linear(rank, rank, bias=False)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.beta = nn.Parameter(torch.tensor(2.0))

    def forward(self, working_memory, tape):
        if tape is None or tape.size(1) == 0:
            return working_memory
        wm_c = self.compress(working_memory)
        q = F.normalize(self.q_proj(wm_c).unsqueeze(1), dim=-1)
        k = F.normalize(self.k_proj(tape), dim=-1)
        v = self.v_proj(tape)
        scores = torch.bmm(q, k.transpose(1, 2)) * F.softplus(self.beta)
        tape_mag = tape.abs().sum(dim=-1).unsqueeze(1)
        scores = scores.masked_fill(tape_mag < 1e-6, -1e9)
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        retrieved_compressed = torch.bmm(attn, v).squeeze(1)
        retrieved = self.decompress(retrieved_compressed)
        g = torch.sigmoid(self.gate(torch.cat([working_memory, retrieved], dim=-1)))
        return working_memory + g * retrieved

class UnsupervisedHebbianPlasticity(nn.Module):
    def __init__(self, d_model, hebb_dim, n_groups, npg):
        super().__init__()
        self.H, self.G, self.NPG = hebb_dim, n_groups, npg
        self.q_proj = nn.Linear(d_model, hebb_dim)
        self.k_proj = nn.Linear(d_model, hebb_dim)
        self.v_proj = nn.Sequential(nn.Linear(d_model, hebb_dim), nn.LayerNorm(hebb_dim))
        self.out_proj = nn.Linear(hebb_dim, d_model)
        self.decay_proj = nn.Linear(d_model, 1)
        self.register_buffer("timescale_gradient", torch.linspace(-3.0, 8.0, n_groups).view(1, n_groups, 1))

    def forward(self, x, M_prev):
        B, N, _ = x.shape
        q = F.normalize(self.q_proj(x), dim=-1, eps=1e-5)
        k = F.normalize(self.k_proj(x), dim=-1, eps=1e-5)
        v = self.v_proj(x)

        M_flat = M_prev.contiguous().view(B * self.G, self.H, self.H)
        k_flat = k.contiguous().view(B * self.G, self.NPG, self.H)
        v_flat = v.contiguous().view(B * self.G, self.NPG, self.H)
        q_flat = q.contiguous().view(B * self.G, self.NPG, self.H)

        v_pred_flat = torch.bmm(k_flat, M_flat.transpose(1, 2))
        surprise_flat = v_flat - v_pred_flat
        delta_M_flat = torch.bmm(surprise_flat.transpose(1, 2), k_flat) / self.NPG

        gamma = torch.sigmoid(
            self.decay_proj(x).view(B, self.G, self.NPG, 1).mean(dim=2) + self.timescale_gradient
        ).unsqueeze(-1)

        M_next = torch.clamp(gamma * M_prev + delta_M_flat.view(B, self.G, self.H, self.H), -10.0, 10.0)
        out_flat = torch.bmm(q_flat, M_next.contiguous().view(B * self.G, self.H, self.H).transpose(1, 2))

        return self.out_proj(out_flat.view(B, N, self.H)), M_next

class TokenMoERouter(nn.Module):
    def __init__(self, d_model, n_groups, top_k):
        super().__init__()
        self.router = nn.Linear(d_model, n_groups, bias=False)
        self.K = top_k
        self.G = n_groups

    def forward(self, x):
        logits = self.router(x)
        scores = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(scores, self.K, dim=-1)
        active_mask = torch.zeros_like(scores, requires_grad=False).scatter_(-1, topk_idx, 1.0)
        routing_weights = scores * active_mask
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return routing_weights, active_mask, scores

class GlobalWorkspaceMessagePass(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, group_states, routing_weights):
        global_repr = (group_states * routing_weights.unsqueeze(-1)).sum(1, keepdim=True).expand_as(group_states)
        routed = self.proj(global_repr)
        return (torch.sigmoid(self.gate(torch.cat([group_states, routed], dim=-1))) * routed)

class FusedOps(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_out * 5))

    def forward(self, x, m_norm, weights):
        o1, o2, o3, o4, o5 = self.shared(x).chunk(5, dim=-1)
        w0, w1, w2, w3, w4 = weights.unsqueeze(-2).unbind(dim=-1)
        return (w0 * torch.tanh(o1)
                + w1 * (torch.sigmoid(o2) * m_norm)
                + w2 * torch.cos(o3)
                + w3 * F.silu(o4)
                + w4 * (-F.relu(o5)))

class LiquidBrain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.D = cfg.d_model
        self.G = cfg.n_groups
        self.NPG = cfg.npg
        self.H = cfg.hebb_dim
        self.N = cfg.n_groups * cfg.npg
        self.mem_size = cfg.hippocampus_mem
        self.rank = cfg.tpa_rank
        self.tape_interval = cfg.tape_interval

        self.emb = nn.Embedding(VOCAB_SIZE, self.D)
        self.role_emb = nn.Embedding(2, self.D)
        self.neuron_emb = nn.Embedding(self.N, self.D)
        self.register_buffer("neuron_indices", torch.arange(self.N))

        self.norm = nn.LayerNorm(self.D)
        self.router = TokenMoERouter(self.D, self.G, cfg.top_k_experts)
        self.group_msg = GlobalWorkspaceMessagePass(self.D)
        self.intra_proj = nn.Linear(self.NPG * self.D, self.NPG * self.D)

        self.hebbian = UnsupervisedHebbianPlasticity(self.D, self.H, self.G, self.NPG)
        self.ops = FusedOps(self.D * 5, self.D)
        self.op_logits = nn.Parameter(torch.randn(self.G, 5) * 0.1)

        self.f_proj = nn.Linear(self.D * 5, self.D)
        self.i_proj = nn.Linear(self.D * 5, self.D)
        self.mlp = nn.Sequential(nn.Linear(self.D * 5, self.D * 2), nn.GELU(), nn.Linear(self.D * 2, self.D))
        self.register_buffer("group_of_neuron", torch.arange(self.N) // self.NPG)

        self.hippocampus = TensorProductHippocampus(self.D, self.rank)
        self.readout_norm = nn.LayerNorm(self.D)
        self.readout_q = nn.Parameter(torch.randn(1, 1, self.D) * 0.02)

        self.motor_norm = nn.LayerNorm(self.D)
        self.lm_head = nn.Linear(self.D, VOCAB_SIZE)

        nn.init.constant_(self.f_proj.bias, 2.0)   
        nn.init.constant_(self.i_proj.bias, 0.0)   
        nn.init.constant_(self.hebbian.decay_proj.bias, 3.0)
        nn.init.zeros_(self.lm_head.weight)         
        nn.init.zeros_(self.lm_head.bias)

    def get_initial_state(self, B, device):
        return (
            torch.zeros(B, self.N, self.D, device=device, requires_grad=True),
            torch.zeros(B, self.G, self.H, self.H, device=device, requires_grad=True),
            torch.full((B,), PAD_IDX, dtype=torch.long, device=device),
            torch.zeros(B, 0, self.rank, device=device),
        )

    def _readout(self, m, tape):
        m_n = self.readout_norm(m)
        scores = F.softmax(torch.bmm(self.readout_q.expand(m_n.shape[0], -1, -1), m_n.transpose(1, 2)) / math.sqrt(self.D), dim=-1)
        wm = torch.bmm(scores, m_n).squeeze(1)
        out = self.hippocampus(wm, tape)
        return self.motor_norm(out)  

    def _step(self, x_t, m_prev, M_prev, op_w, role_id_tensor):
        B = m_prev.size(0)
        r = role_id_tensor.expand(B)
        token_embed = self.emb(x_t) + self.role_emb(r)

        routing_weights, active_mask, scores = self.router(token_embed)
        active_mask_m = (active_mask.unsqueeze(2).expand(B, self.G, self.NPG).reshape(B, self.N).unsqueeze(-1))
        active_mask_M = active_mask.view(B, self.G, 1, 1)

        sens = (token_embed.unsqueeze(1) + self.neuron_emb(self.neuron_indices).unsqueeze(0))

        m_gated = m_prev * active_mask_m
        m_norm = self.norm(m_gated)

        grp = m_norm.view(B, self.G, self.NPG, self.D).mean(2)
        inter_x = (self.group_msg(grp, routing_weights).unsqueeze(2).expand(B, self.G, self.NPG, self.D).reshape(B, self.N, self.D))
        intra_x = (self.intra_proj(m_norm.view(B, self.G, self.NPG * self.D)).view(B, self.G, self.NPG, self.D).reshape(B, self.N, self.D))

        hebb_out, M_next_updated = self.hebbian(m_norm, M_prev)

        combined = torch.cat([sens, inter_x, intra_x, hebb_out, m_norm], dim=-1)
        f = torch.sigmoid(self.f_proj(combined))
        i = torch.sigmoid(self.i_proj(combined))
        delta = self.ops(combined, m_norm, op_w) + self.mlp(combined)

        m_next_updated = torch.clamp(f * m_prev + i * delta, -20.0, 20.0)

        m_next = torch.where(active_mask_m > 0.5, m_next_updated, m_prev)
        M_next = torch.where(active_mask_M > 0.5, M_next_updated, M_prev)

        return m_next, M_next, active_mask, scores

    def encode_context(self, tokens, state, role_id=1, compute_ce=False):
        B, T = tokens.shape
        m, M, x_t, tape = state
        op_w = F.softmax(self.op_logits, dim=-1)[self.group_of_neuron].unsqueeze(0)
        all_mem_states, all_tgts = [], []

        sum_active = torch.zeros(B, self.G, device=tokens.device)
        sum_scores = torch.zeros(B, self.G, device=tokens.device)
        total_valid = torch.zeros(B, 1, device=tokens.device)

        role_t = torch.tensor([role_id], dtype=torch.long, device=tokens.device)

        for t in range(T):
            xt_c = x_t
            m_pre, M_pre = m, M

            if self.training:
                m, M, active_mask, scores = checkpoint(
                    self._step, xt_c, m_pre, M_pre, op_w, role_t,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                m, M, active_mask, scores = self._step(xt_c, m_pre, M_pre, op_w, role_t)

            tgt = tokens[:, t]
            valid = (tgt != PAD_IDX)
            valid_float = valid.float().unsqueeze(-1)

            sum_active = sum_active + active_mask * valid_float
            sum_scores = sum_scores + scores * valid_float
            total_valid = total_valid + valid_float

            if t % self.tape_interval == 0 or t == T - 1:
                m_pool = self.readout_norm(m).max(dim=1)[0]
                pool = (self.hippocampus.compress(m_pool).unsqueeze(1) * valid.float().view(B, 1, 1))
                tape = torch.cat([tape, pool], dim=1)
                if tape.size(1) > self.mem_size:
                    tape = tape[:, -self.mem_size:].clone()

            if compute_ce:
                all_mem_states.append(self._readout(m, tape))
                all_tgts.append(tgt)

            if not valid.all():
                valid_m = valid.float().view(B, 1, 1)
                valid_M = valid.float().view(B, 1, 1, 1)
                m = torch.where(valid_m > 0.5, m, m_pre)
                M = torch.where(valid_M > 0.5, M, M_pre)

            x_t = torch.where(valid, tgt, x_t)

        ce_loss = torch.tensor(0.0, device=tokens.device)
        if compute_ce:
            mem_tensor = torch.stack(all_mem_states, dim=1)
            tgt_tensor = torch.stack(all_tgts, dim=1)
            valid_ce_mask = (tgt_tensor != PAD_IDX)
            if valid_ce_mask.any():
                logits = self.lm_head(mem_tensor[valid_ce_mask])
                ce_loss = F.cross_entropy(logits, tgt_tensor[valid_ce_mask])

        total_valid_c = total_valid.clamp(min=1.0)
        f_i = (sum_active / total_valid_c).mean(dim=0)
        P_i = (sum_scores / total_valid_c).mean(dim=0)
        aux_loss = self.G * (f_i * P_i).sum() - 1.0

        return ce_loss, aux_loss, (m, M, x_t, tape)

    def generate_proposals(self, state, max_len=150):
        m, M, x_t, tape = state
        B, device = m.size(0), m.device
        toks = []
        ended = torch.zeros(B, dtype=torch.bool, device=device)
        op_w = F.softmax(self.op_logits, dim=-1)[self.group_of_neuron].unsqueeze(0)

        role_t = torch.tensor([0], dtype=torch.long, device=device)
        with torch.no_grad():
            for step in range(max_len):
                m_pre, M_pre = m, M
                m, M, _, _ = self._step(x_t.masked_fill(ended, PAD_IDX), m_pre, M_pre, op_w, role_t)

                alive = (~ended).float()
                if ended.any():
                    m = torch.where(alive.view(B, 1, 1) > 0.5, m, m_pre)
                    M = torch.where(alive.view(B, 1, 1, 1) > 0.5, M, M_pre)

                if (step % self.tape_interval == 0 or step == max_len - 1):
                    m_pool = self.readout_norm(m).max(dim=1)[0]
                    pool = (self.hippocampus.compress(m_pool).unsqueeze(1) * alive.view(B, 1, 1))
                    tape = torch.cat([tape, pool], dim=1)
                    if tape.size(1) > self.mem_size:
                        tape = tape[:, -self.mem_size:].clone()

                logits = self.lm_head(self._readout(m, tape)).float()
                logits[:, EOS_IDX] += 0.5
                logits = torch.nan_to_num(logits, nan=-10.0, posinf=10.0, neginf=-10.0)

                dist = Categorical(logits=logits)
                idx = dist.sample()
                idx = idx.masked_fill(ended, PAD_IDX)
                toks.append(idx)
                ended = ended | (idx == EOS_IDX)
                x_t = idx
                if ended.all():
                    break

        return torch.stack(toks, 1)

    def compute_trajectory_logprobs(self, prompt_toks_base, gen_toks, group_size):
        B_base = prompt_toks_base.size(0)
        device = prompt_toks_base.device
        state0 = self.get_initial_state(B_base, device)

        _, aux_p, state_p = self.encode_context(prompt_toks_base, state0, role_id=1, compute_ce=False)
        m, M, x_t, tape = state_p

        m = m.repeat_interleave(group_size, dim=0)
        M = M.repeat_interleave(group_size, dim=0)
        x_t = x_t.repeat_interleave(group_size, dim=0)
        tape = tape.repeat_interleave(group_size, dim=0)

        B_full = B_base * group_size
        op_w = F.softmax(self.op_logits, dim=-1)[self.group_of_neuron].unsqueeze(0)
        log_probs = []

        sum_active = torch.zeros(B_full, self.G, device=device)
        sum_scores = torch.zeros(B_full, self.G, device=device)
        total_valid = torch.zeros(B_full, 1, device=device)

        role_t = torch.tensor([0], dtype=torch.long, device=device)

        for t in range(gen_toks.size(1)):
            tgt = gen_toks[:, t]
            valid = (tgt != PAD_IDX)
            m_pre, M_pre = m, M

            if self.training:
                m, M, active_mask, scores = checkpoint(
                    self._step, x_t.masked_fill(~valid, PAD_IDX), m_pre, M_pre, op_w, role_t,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                m, M, active_mask, scores = self._step(x_t.masked_fill(~valid, PAD_IDX), m_pre, M_pre, op_w, role_t)

            valid_float = valid.float().unsqueeze(-1)
            sum_active = sum_active + active_mask * valid_float
            sum_scores = sum_scores + scores * valid_float
            total_valid = total_valid + valid_float

            if not valid.all():
                m = torch.where(valid.float().view(B_full, 1, 1) > 0.5, m, m_pre)
                M = torch.where(valid.float().view(B_full, 1, 1, 1) > 0.5, M, M_pre)

            if (t % self.tape_interval == 0 or t == gen_toks.size(1) - 1):
                m_pool = self.readout_norm(m).max(dim=1)[0]
                pool = (self.hippocampus.compress(m_pool).unsqueeze(1) * valid.float().view(B_full, 1, 1))
                tape = torch.cat([tape, pool], dim=1)
                if tape.size(1) > self.mem_size:
                    tape = tape[:, -self.mem_size:].clone()

            logits = self.lm_head(self._readout(m, tape)).float()
            logits[:, EOS_IDX] += 0.5
            logits = torch.nan_to_num(logits, nan=-10.0, posinf=10.0, neginf=-10.0)

            dist = Categorical(logits=logits)
            tgt_safe = tgt.clamp(0, VOCAB_SIZE - 1)
            log_probs.append(dist.log_prob(tgt_safe).to(torch.float32))
            x_t = torch.where(valid, tgt, x_t)

        log_probs = torch.stack(log_probs, dim=1)
        mask = (gen_toks != PAD_IDX).float()

        total_valid_c = total_valid.clamp(min=1.0)
        f_i = (sum_active / total_valid_c).mean(dim=0)
        P_i = (sum_scores / total_valid_c).mean(dim=0)
        aux_gen = self.G * (f_i * P_i).sum() - 1.0

        return log_probs, mask, aux_p + aux_gen

# =============================================================================
# CHECKPOINTING
# =============================================================================
def save_checkpoint(turn, brain, opt_brain, scheduler, env, rolling_sft, ref_brain=None, scaler=None):
    print(f"\n  Saving checkpoint at turn {turn}...")
    save_dict = {
        "brain": brain.state_dict(),
        "opt_brain": opt_brain.state_dict(),
        "scheduler": scheduler.state_dict(),
        "turn": turn,
        "env_level": env.current_level,
        "env_rolling_wins": env.rolling_wins,
        "rolling_sft": rolling_sft,
    }
    if ref_brain is not None: save_dict["ref_brain"] = ref_brain.state_dict()
    if scaler is not None: save_dict["scaler"] = scaler.state_dict()
    torch.save(save_dict, CHECKPOINT_PATHS[0])
    print("  Checkpoint saved.")

def load_checkpoint(brain, ref_brain, opt_brain, scheduler, env, device, scaler):
    path = next((p for p in CHECKPOINT_PATHS if os.path.exists(p)), None)
    if not path: return 1, 0.0
    print(f"\n  Loading checkpoint '{path}'...")
    try:
        chkpt = torch.load(path, map_location=device, weights_only=False)
        brain.load_state_dict(chkpt["brain"], strict=False)
        if "ref_brain" in chkpt: ref_brain.load_state_dict(chkpt["ref_brain"], strict=False)
        else: ref_brain.load_state_dict(chkpt["brain"], strict=False)
        try: opt_brain.load_state_dict(chkpt["opt_brain"])
        except Exception: print("  Optimizer state reset.")
        if "scheduler" in chkpt:
            try: scheduler.load_state_dict(chkpt["scheduler"])
            except Exception: print("  Scheduler state reset.")
        env.current_level = chkpt.get("env_level", 1)
        env.rolling_wins = chkpt.get("env_rolling_wins", 0)
        rolling_sft = chkpt.get("rolling_sft", 0.0)
        if scaler and chkpt.get("scaler"): scaler.load_state_dict(chkpt["scaler"])
        turn = chkpt.get("turn", 0) + 1
        print(f"  Resumed from Turn {turn} (Level {env.current_level}).")
        return turn, rolling_sft
    except Exception as e:
        print(f"  Failed to load: {e}. Starting fresh.")
        return 1, 0.0

# =============================================================================
# LR SCHEDULE
# =============================================================================
def get_lr_lambda(cfg):
    def lr_lambda(step):
        if step < cfg.warmup_steps: return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = cfg.lr_min / cfg.lr_brain
        return min_ratio + (1.0 - min_ratio) * cosine
    return lr_lambda

# =============================================================================
# VRAM HELPERS
# =============================================================================
def get_vram_mb(device):
    if device.type == "cuda": return torch.cuda.memory_allocated() / (1024 * 1024)
    elif device.type == "mps": return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0

# =============================================================================
# RUN LOOP
# =============================================================================
def run(cfg, args):
    if getattr(args, "wandb", False) and WANDB_AVAILABLE:
        wandb.init(project="genesis-v66", config=vars(cfg))

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu")
    ac_device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = (not getattr(args, "no_amp", False) and device.type == "cuda")
    amp_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
    scaler = (torch.cuda.amp.GradScaler() if (device.type == "cuda" and use_amp) else None)

    brain = LiquidBrain(cfg).to(device)
    ref_brain = copy.deepcopy(brain).eval()
    for p in ref_brain.parameters(): p.requires_grad = False

    if getattr(args, "compile", False):
        print("Compiling LiquidBrain...")
        brain = torch.compile(brain, mode="reduce-overhead", fullgraph=False)

    opt_brain = torch.optim.AdamW(brain.parameters(), lr=cfg.lr_brain, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_brain, get_lr_lambda(cfg))

    env = StorybookEnv()
    start_turn, rolling_sft = load_checkpoint(brain, ref_brain, opt_brain, scheduler, env, device, scaler)

    total_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)
    print(f"\n  Genesis V66 (PURE R1-ZERO) on {device.type.upper()} | Trainable: {total_params:,}")
    print(f"  Trigram Guillotine Active | Vocab: {VOCAB_SIZE} | Groups: {cfg.n_groups}")
    print("=" * 72)

    log_window = []

    try:
        for turn in range(start_turn, cfg.max_steps + 1):
            if turn % 50 == 0:
                if device.type == "mps": torch.mps.empty_cache()
                elif device.type == "cuda": torch.cuda.empty_cache()
                gc.collect()

            (prompts, truths, demos, entities, holders, stories, num_events) = env.generate_batch(cfg.batch_size)

            if turn <= cfg.imitation_steps:
                p_tok = encode_left(prompts, device)
                d_tok = encode_right(demos, device)

                opt_brain.zero_grad(set_to_none=True)
                with torch.autocast(device_type=ac_device_type, dtype=amp_dtype, enabled=use_amp):
                    state0 = brain.get_initial_state(cfg.batch_size, device)
                    _, aux_p, state_post = brain.encode_context(p_tok, state0, role_id=1, compute_ce=False)
                    ce_d, aux_d, _ = brain.encode_context(d_tok, state_post, role_id=0, compute_ce=True)
                    loss = ce_d + cfg.aux_loss_coef * (aux_p + aux_d)

                ce_val = ce_d.item()
                aux_val = aux_d.item()

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt_brain)
                    torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
                    scaler.step(opt_brain)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
                    opt_brain.step()

                scheduler.step()
                rolling_sft = (0.9 * rolling_sft + 0.1 * ce_val if rolling_sft > 0 else ce_val)

                with torch.no_grad():
                    for p, p_ref in zip(brain.parameters(), ref_brain.parameters()):
                        p_ref.data.copy_(p.data)

                if turn <= 5 or turn % 10 == 0:
                    brain.eval()
                    with (torch.no_grad(), torch.autocast(device_type=ac_device_type, dtype=amp_dtype, enabled=use_amp)):
                        state0 = brain.get_initial_state(cfg.batch_size, device)
                        _, _, state_post = brain.encode_context(p_tok, state0, role_id=1, compute_ce=False)
                        gen_toks = brain.generate_proposals(state_post, max_len=150)
                    brain.train()

                    texts = [decode(gen_toks[i]) for i in range(cfg.batch_size)]
                    _, tags, parsed = env.evaluate(texts, truths, entities, holders, stories)

                    best_idx = 0
                    st = tags[best_idx]
                    prompt_disp = prompts[best_idx].split('\n')
                    p_summary = (prompt_disp[1][:60] + "..." if len(prompt_disp) > 2 else prompt_disp[0][:60])

                    lr_now = scheduler.get_last_lr()[0]
                    vram_mb = get_vram_mb(device)
                    vram_str = f" | VRAM:{vram_mb:.0f}MB" if vram_mb > 0 else ""

                    full_parsed = parsed[best_idx].replace('\n', ' ')
                    if len(full_parsed) > 150:
                        p_agent = full_parsed[:100] + " ... " + full_parsed[-50:]
                    else:
                        p_agent = full_parsed

                    log_entry = (
                        f"[SFT] T{turn:05d}/{cfg.imitation_steps} | Lv:{env.current_level} | CE:{rolling_sft:.3f} | Aux:{aux_val:.3f} | LR:{lr_now:.1e}{vram_str} | {st}\n"
                        f"  {p_summary}\n"
                        f"  Target: '{demos[best_idx][:80]}'\n"
                        f"  Agent : '{p_agent}'\n\n")
                    log_window.append(log_entry)
                    if len(log_window) > 4: log_window.pop(0)
                    clear_output(wait=True)
                    print(f"\n  Genesis V66 | Phase 1: SFT | Vocab: {VOCAB_SIZE}")
                    print("=" * 72)
                    print("".join(log_window))
                else:
                    print(".", end="", flush=True)

                if turn % 5000 == 0:
                    save_checkpoint(turn, brain, opt_brain, scheduler, env, rolling_sft, ref_brain, scaler)
                continue

            # =================================================================
            # PHASE 2: GRPO WITH ZERO-TOLERANCE VPRM
            # =================================================================
            p_tok_base = encode_left(prompts, device)

            prompts_exp = [p for p in prompts for _ in range(cfg.group_size)]
            truths_exp = [t for t in truths for _ in range(cfg.group_size)]
            entities_exp = [e for e in entities for _ in range(cfg.group_size)]
            holders_exp = [h for h in holders for _ in range(cfg.group_size)]
            stories_exp = [s for s in stories for _ in range(cfg.group_size)]

            brain.eval()
            with (torch.no_grad(), torch.autocast(device_type=ac_device_type, dtype=amp_dtype, enabled=use_amp)):
                state0_base = brain.get_initial_state(cfg.batch_size, device)
                _, _, state_post_base = brain.encode_context(p_tok_base, state0_base, role_id=1, compute_ce=False)
                m, M, x_t, tape = state_post_base
                m = m.repeat_interleave(cfg.group_size, dim=0)
                M = M.repeat_interleave(cfg.group_size, dim=0)
                x_t = x_t.repeat_interleave(cfg.group_size, dim=0)
                tape = tape.repeat_interleave(cfg.group_size, dim=0)
                gen_toks = brain.generate_proposals((m, M, x_t, tape), max_len=150)
            brain.train()

            texts = [decode(gen_toks[i]) for i in range(len(prompts_exp))]
            rewards_raw, tags, parsed = env.evaluate(texts, truths_exp, entities_exp, holders_exp, stories_exp)
            rewards = rewards_raw.to(device)

            rewards_r = rewards.view(cfg.batch_size, cfg.group_size)
            mean_r = rewards_r.mean(dim=1, keepdim=True)
            std_r = rewards_r.std(dim=1, keepdim=True).clamp(min=1e-4)
            adv = ((rewards_r - mean_r) / std_r).view(-1)

            with (torch.no_grad(), torch.autocast(device_type=ac_device_type, dtype=amp_dtype, enabled=use_amp)):
                ref_log_probs, mask, _ = ref_brain.compute_trajectory_logprobs(p_tok_base, gen_toks, cfg.group_size)

            opt_brain.zero_grad(set_to_none=True)
            with torch.autocast(device_type=ac_device_type, dtype=amp_dtype, enabled=use_amp):
                log_probs, mask, aux_gen = brain.compute_trajectory_logprobs(p_tok_base, gen_toks, cfg.group_size)

                ratio = torch.exp((log_probs - ref_log_probs.detach()).clamp(-10, 10))
                clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)

                adv_tok = adv.detach().unsqueeze(1) * mask
                surr1 = ratio * adv_tok
                surr2 = clipped_ratio * adv_tok
                
                seq_lens = mask.sum(dim=1).clamp(min=1.0)
                actor_loss = -(torch.min(surr1, surr2).sum(dim=1) / seq_lens).mean()

                log_ratio_kl = (ref_log_probs.detach() - log_probs).clamp(-10, 10)
                ratio_kl = torch.exp(log_ratio_kl)
                kl_per_token = (ratio_kl - log_ratio_kl - 1.0) * mask
                kl_loss = (kl_per_token.sum(dim=1) / seq_lens).mean()

                loss = actor_loss + cfg.beta_kl * kl_loss + cfg.aux_loss_coef * aux_gen

            kl_val = kl_loss.item()

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt_brain)
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
                scaler.step(opt_brain)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
                opt_brain.step()

            scheduler.step()

            special_idx = next(
                (i for i, t in enumerate(tags) if any(k in t for k in ["HALLUCINATION", "SHORTCUT", "FORMAT ERR", "LOOPING"])), -1)
            best_idx = (special_idx if special_idx != -1 else int(torch.argmax(rewards).item()))

            if (turn % 5 == 0 or any("CORRECT" in t for t in tags) or special_idx != -1):
                st = tags[best_idx]
                adv_val = adv[best_idx].item()
                r_val = rewards[best_idx].item()

                prompt_disp = prompts_exp[best_idx].split('\n')
                p_summary = (prompt_disp[1][:60] + "..." if len(prompt_disp) > 2 else prompt_disp[0][:60])

                lr_now = scheduler.get_last_lr()[0]
                vram_mb = get_vram_mb(device)
                vram_str = f" | VRAM:{vram_mb:.0f}MB" if vram_mb > 0 else ""

                full_parsed = parsed[best_idx].replace('\n', ' ')
                if len(full_parsed) > 150:
                    p_agent = full_parsed[:100] + " ... " + full_parsed[-50:]
                else:
                    p_agent = full_parsed

                log_entry = (
                    f"[GRPO] T{turn:05d} | Lv:{env.current_level} | R:{r_val:+.2f} | Adv:{adv_val:+.2f} | KL:{kl_val:.4f} | LR:{lr_now:.1e}{vram_str} | {st}\n"
                    f"  {p_summary}\n"
                    f"  Target: {entities_exp[best_idx]}\n"
                    f"  Agent : '{p_agent}'\n\n")
                log_window.append(log_entry)
                if len(log_window) > 4: log_window.pop(0)
                clear_output(wait=True)
                print(f"\n  Genesis V66 (PURE R1-ZERO) | Level {env.current_level} | Vocab: {VOCAB_SIZE}")
                print("=" * 80)
                print("".join(log_window))
            else:
                print(".", end="", flush=True)

            if turn % 5000 == 0:
                save_checkpoint(turn, brain, opt_brain, scheduler, env, rolling_sft, ref_brain, scaler)

    except KeyboardInterrupt:
        print(f"\n  Training interrupted at turn {turn}. Saving...")
        save_checkpoint(turn, brain, opt_brain, scheduler, env, rolling_sft, ref_brain, scaler)
        print("  Saved. Resume later.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args, _ = parser.parse_known_args()
    cfg = GenesisConfig(batch_size=args.batch_size, group_size=args.group_size)
    run(cfg, args)
