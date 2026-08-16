---
name: game-engine-dev
description: >-
  Use this skill when you need to create, modify, or debug the Ulti card game engine. 
  It provides guidelines on how the environment states should be represented, how the 
  action step functions should operate, and how to keep it compatible with the RL agent.
---

# Game Engine Development Skill

Use this skill to build out the Gymnasium-compatible simulation engine for the card game **Ulti**.

## Core Principles
1. **Card Game State Representation**: The state must represent hands, the talon, trick history, trump suit, and the current phase (bidding or playing). This should be a combination of discrete variables and boolean arrays.
2. **Action Masking**: Crucial for card games. The engine *must* output an `action_mask` alongside the observation, indicating which cards are legal to play or which bids are legal.
3. **OpenAI Gym / Gymnasium Compatibility**: Ensure the engine exposes a `step(action)` and `reset()` interface.
4. **Performance First**: Vectorize state generations using NumPy where applicable so that running thousands of simulations for training does not bottleneck the agent.

## Workflow
1. When asked to create a new game environment, place it in the `engine/environments/` directory.
2. Inherit from `gymnasium.Env`.
3. Define the `observation_space` (e.g., `Dict` of `MultiBinary` or `Discrete`) and `action_space` (e.g., `Discrete` representing the 32 cards or possible bids).
4. Implement `reset()` to deal cards, set the talon, and return the initial observation and info dict (which includes `action_mask`).
5. Implement `step(action)` to transition the state (bid, discard, or play a card), compute the reward, and determine if the episode is terminated or truncated.
