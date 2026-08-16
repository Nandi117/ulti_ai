---
name: neuro-symbolic-rl
description: >-
  Use this skill when developing, tuning, or debugging the Neuro-Symbolic Reinforcement 
  Learning agent using PyTorch. It explains the multi-level architecture integrating 
  neural networks with symbolic logic for the Ulti card game.
---

# Neuro-Symbolic RL Agent Skill

Use this skill to work on the PyTorch-based agent that combines deep learning with symbolic reasoning for the complex card game **Ulti**.

## Architecture Overview
The agent uses a multi-level architecture:
1. **Neural Level (Perception & Intuition)**: MLPs or Transformers (in PyTorch) that process the observed game state (hands, trick history) to extract sub-symbolic features and estimate value/policy.
2. **Symbolic Level (Reasoning & Rules)**: A logic module that takes the exact rules of Ulti (e.g., "must follow suit", "Ulti bid requires winning last trick with 7 of trumps") and applies them to constrain the neural network's actions via action masking, or to explicitly deduce hidden information (card counting).
3. **Action Level**: Combines the learned policy with the symbolic constraints to execute a valid card play or bid.

## Development Guidelines
1. **PyTorch Integration**: All neural components must be developed as `torch.nn.Module`.
2. **Symbolic Module**: Implement the trick-taking logic cleanly in `agent/symbolic/rules.py`.
3. **Action Masking**: The neural network's output logits MUST be masked by the symbolic logic module before the softmax/sampling step to guarantee legal plays.
4. **State Handling**: The engine outputs flat arrays or dictionaries. The neural part must process these embeddings.
