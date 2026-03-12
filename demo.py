"""
DigitalBaby: Genesis V66 Demo Script

This script loads a trained Genesis V66 checkpoint and demonstrates 
the emergent reasoning of the LiquidBrain architecture.
"""

import torch
import argparse
import os
from genesis_v66 import LiquidBrain, GenesisConfig, StorybookEnv, encode_left, decode

def run_demo(checkpoint_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Genesis V66 on {device}...")

    # Load config and environment
    cfg = GenesisConfig()
    env = StorybookEnv()
    
    # Initialize model
    brain = LiquidBrain(cfg).to(device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        return

    try:
        chkpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        brain.load_state_dict(chkpt["brain"], strict=False)
        print(f"Successfully loaded checkpoint from turn {chkpt.get('turn', 'unknown')}.")
        env.current_level = chkpt.get("env_level", 1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    brain.eval()
    
    # Generate a sample from the environment
    print("\n" + "="*80)
    print("GENERATING STORY PROMPT")
    print("="*80)
    
    (prompts, truths, _, entities, holders, stories, _) = env.generate_batch(1)
    prompt = prompts[0]
    truth = truths[0]
    entity = entities[0]
    
    print(prompt.replace("👶", ""))
    
    # Encode and generate
    p_tok = encode_left([prompt], device)
    
    with torch.no_grad():
        state0 = brain.get_initial_state(1, device)
        _, _, state_post = brain.encode_context(p_tok, state0, role_id=1, compute_ce=False)
        gen_toks = brain.generate_proposals(state_post, max_len=150)
        
    response = decode(gen_toks[0])
    
    print("\n" + "="*80)
    print("AGENT REASONING (EMERGENT)")
    print("="*80)
    print(response)
    print("-" * 80)
    print(f"Target: {truth}")
    
    # Simple check
    if truth.lower() in response.lower():
        print("RESULT: ✅ CORRECT")
    else:
        print("RESULT: ❌ INCORRECT")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="genesis_v66_checkpoint.pt", help="Path to the trained model checkpoint.")
    args = parser.parse_args()
    
    run_demo(args.checkpoint)
