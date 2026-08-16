---
name: neuro-symbolic-rl
description: >-
  Use this skill when developing, tuning, or debugging the Neuro-Symbolic Reinforcement 
  Learning agent using PyTorch. It explains the multi-level architecture integrating 
  neural networks with symbolic logic.
---

# Neuro-Symbolic RL Agent Skill

Use this skill to work on the PyTorch-based agent that combines deep learning with symbolic reasoning.

## Architecture Overview
The agent uses a multi-level architecture:
1. **Neural Level (Perception)**: CNNs or MLPs (in PyTorch) that process raw grid states to extract sub-symbolic features.
2. **Symbolic Level (Reasoning)**: A logic module that takes the extracted features and applies symbolic rules (e.g., Propositional/First-Order Logic) to make high-level decisions or constrain the neural network's actions.
3. **Action Level**: Combines the learned policy with the symbolic constraints to execute an action in the environment.

## Development Guidelines
1. **PyTorch Integration**: All neural components must be developed as `torch.nn.Module`.
2. **Symbolic Module**: Implement the logic system clearly. If you are using a neuro-symbolic framework or building a custom logic parser, keep it separated in `agent/symbolic/`.
3. **Training Loop**: The main RL loop (e.g., PPO or DQN) should live in `agent/training/` and cleanly interact with the engine's `step()` function.
4. **State Handling**: The engine outputs grid data. The neural part might process the grid as a tensor of shape `(Channels, Width, Height)`. The symbolic part requires parsing this into entities (e.g., `at(player, x, y)`).

## Workflow
1. When adjusting the network, modify `agent/neural/models.py`.
2. When adjusting rules or logical constraints, modify `agent/symbolic/rules.py`.
3. Keep training metrics (reward, episode length, logic violations) logged using tools like TensorBoard or Weights & Biases.
