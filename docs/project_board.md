# Rablóulti AI - Project Board

## Phase 1: Planning & Architecture (COMPLETED)
- [x] **Task 1**: Implement `Card` and `Deck` primitives for the 32-card German deck.
- [x] **Task 2**: Map out the mathematical representation of the Bidding Action Space (MultiDiscrete vs Flat Discrete) to handle the 50+ Rablóulti combinations.
- [x] **Task 3**: Map out the Observation Space for the neural network (Hand, Talon, Trick History, Current Bid, Bid Hierarchy).
- [x] **Task 4**: Design the Action Masking logic algorithm (Bid hierarchy enforcement, Suit-following enforcement).

## Phase 2: Engine Implementation (CURRENT)
- [ ] **Task 5**: Implement the Rablóulti Bidding Phase logic.
- [ ] **Task 6**: Implement the Rablóulti Trick-taking Phase logic.
- [ ] **Task 7**: Wrap the engine in a `gymnasium.Env`.

## Phase 3: Neuro-Symbolic Agent Integration (PAUSED)
- [ ] **Task 8**: Symbolic module to strictly enforce action masking.
- [ ] **Task 9**: PPO Training loop setup.
