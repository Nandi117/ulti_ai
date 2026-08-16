# Rablóulti AI - Architectural Design Document

This document outlines the core mathematical and architectural decisions for the neuro-symbolic RL agent and the game engine, as established during the Phase 1 planning phase.

## 1. Engine Mathematics
- **Card Representation**: Integer IDs (0 to 31). Suit and Rank are extracted via modulo/division math to maximize NumPy vectorization performance.
- **Hand Representation**: A Flat Boolean Array of length 32. Index `i` is 1 if the player holds the card, else 0.
- **Hidden Information (Information Set)**: Explicit Neuro-Symbolic Deduction. The engine's logic module will actively track known facts (e.g. "Player 2 is void in Acorns" or "Player 3 dropped the talon"). These facts are passed as explicit boolean flags in the observation array, rather than forcing the neural network to deduce them purely from chronological history.

## 2. Action Spaces
- **Bidding Phase**: Flat Discrete Space (size ~60). Each integer maps to a specific Rablóulti combination (e.g. `Piros 20-100 ulti`). Action masking simply zeroes out integers whose point values are lower than the current highest bid.
- **Playing Phase**: Flat Discrete Space (size 32). The agent outputs the absolute Integer ID of the card it wishes to play. Action masking zeroes out cards not in the hand, or cards that illegally break the "follow suit" and "trump" rules.

## 3. RL & Reward Architecture
- **Reward Structure**: Terminal Reward Only. The environment outputs 0 intermediate reward, and only calculates the exact point exchange (+48, -96, etc.) on the final step of the hand. This prevents "greedy" trick-taking in games where losing is the goal.
- **Neural Network Architecture**: Multi-headed by Game Family.
  - The base network processes the state embedding.
  - The output is routed to specific policy heads based on the current contract:
    - **Normal/Ulti Head**: Strategy focused on winning tricks, capturing points, and saving the 7 of trumps for the end.
    - **Betli Head**: Strategy focused on losing tricks and avoiding capture.
    - **Durchmars Head**: Strategy focused on absolute control and sweeping all tricks.
  - The separation prevents catastrophic forgetting and confusion caused by contradictory game objectives.
