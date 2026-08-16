# Rablóulti AI - Project Board

## Phase 1: Planning & Architecture (COMPLETED)
- [x] **Task 1**: Implement `Card` and `Deck` primitives for the 32-card German deck.
- [x] **Task 2**: Map out the mathematical representation of the Bidding Action Space (MultiDiscrete vs Flat Discrete) to handle the 50+ Rablóulti combinations.
- [x] **Task 3**: Map out the Observation Space for the neural network (Hand, Talon, Trick History, Current Bid, Bid Hierarchy).
- [x] **Task 4**: Design the Action Masking logic algorithm (Bid hierarchy enforcement, Suit-following enforcement).

## Phase 2: Engine Implementation (COMPLETED)
- [x] **Task 5**: Implement the Rablóulti Bidding Phase logic.
- [x] **Task 6**: Implement the Rablóulti Trick-taking Phase logic.
- [x] **Task 7**: Wrap the engine in a `gymnasium.Env`.

## Phase 3: Neuro-Symbolic Agent Integration (CURRENT)
- [ ] **Task 8**: Implement Masked Softmax logic for PyTorch and a Baseline Heuristic Agent.
- [ ] **Task 9**: Implement the PPO MARL Training Loop with Fictitious Play.
