---
name: game-engine-dev
description: >-
  Use this skill when you need to create, modify, or debug the 2D grid-based game engine. 
  It provides guidelines on how the environment states should be represented, how the 
  action step functions should operate, and how to keep it compatible with the RL agent.
---

# Game Engine Development Skill

Use this skill to build out the 2D grid-based simulation engine in Python.

## Core Principles
1. **Grid-Based State**: The state must be representable as an NxM grid or a dictionary of logical predicates so the Neuro-Symbolic RL agent can interpret it easily.
2. **OpenAI Gym / Gymnasium Compatibility**: Ensure the engine exposes a `step(action)` and `reset()` interface to be fully compatible with standard RL training loops.
3. **Performance First**: Vectorize grid operations using NumPy where applicable so that running thousands of simulations for training does not bottleneck the agent.
4. **Decoupled Logic and Rendering**: The game logic must function entirely independent of the visual rendering. Rendering (via Pygame or Arcade) should only happen when a `render()` method is called.

## Workflow
1. When asked to create a new game environment, place it in the `engine/environments/` directory.
2. Inherit from `gymnasium.Env`.
3. Define the `observation_space` (e.g., `Box` or `MultiDiscrete`) and `action_space` (e.g., `Discrete` or `MultiDiscrete`).
4. Implement `reset()` to initialize/reset the grid state and return the initial observation and info dict.
5. Implement `step(action)` to transition the state, compute the reward, and determine if the episode is terminated or truncated.
6. Provide a `render()` function that visually draws the 2D grid if requested.

## Useful Resources
- Ensure `numpy` is used for grid representations.
- Keep symbolic elements (like "Key", "Door", "Player") easily queryable from the state so the symbolic part of the RL agent can extract rules.
