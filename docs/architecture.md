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

## 4. Multi-Agent RL Training Loop
- **Self-Play Dynamics**: Training begins against a hardcoded Heuristic/Random Baseline agent to verify mathematical stability. It then graduates to Fictitious Play, where the agent plays against a pool of its own historical checkpoints to prevent overfitting.
- **Role Asymmetry (3-Player)**: A Shared Policy approach is used. The network learns to play both the Declarer (solo) and Defender (co-op) roles using the same weights. A boolean flag (`is_declarer=True/False`) is passed into the observation state to context-switch the agent's strategy.
- **Masked Softmax**: The engine's Action Mask is enforced directly inside the PyTorch forward pass. A massive negative penalty (`-1e9`) is added to the logits of illegal moves prior to the Softmax activation, mathematically ensuring illegal moves have a 0.0% probability of being sampled.

## 5. Telemetry & Interactive Training Loop
- **Hot-Reloading Hyperparameters**: To allow the ML Expert Agent and human users to adjust parameters on the fly without killing the long training loop, hyperparameters (Learning Rate, Entropy Coefficient, Batch Size) will be stored in a `C:/ulti_ai/config/hyperparams.json` file. The PyTorch training loop will check this file every N epochs and hot-reload the values.
- **Real-Time Monitoring**: The training script will integrate with TensorBoard to output real-time graphical metrics for human viewing. It will also output a rolling `C:/ulti_ai/logs/telemetry.jsonl` file containing textual summaries of loss, entropy, and win-rates. The `ml_expert_agent` can continuously read this file to evaluate the health of the training and decide whether to alter the `hyperparams.json` file.
