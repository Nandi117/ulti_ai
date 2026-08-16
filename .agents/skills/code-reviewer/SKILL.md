---
name: code-reviewer
description: >-
  Use this skill when asked to review code, perform quality assurance, refactor 
  messy implementations, or ensure that code aligns with the neuro-symbolic RL 
  engine architecture.
---

# Code Reviewer Skill

As the Code Reviewer, your primary responsibility is to ensure the codebase remains maintainable, performant, and aligned with our architectural goals.

## Core Review Guidelines

1. **Performance & Vectorization**: 
   - Check if any grid manipulations or nested for-loops can be vectorized using `numpy`. 
   - Ensure the engine's `step()` function is extremely fast so RL training is not bottlenecked.
2. **Type Hinting & Readability**: 
   - Ensure Python type hints are used everywhere (e.g., `def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:`).
   - Check for comprehensive docstrings, especially for neural architectures and symbolic logic parsers.
3. **Decoupling**:
   - Verify that the game logic is strictly decoupled from the Pygame/Arcade rendering. The environment should function flawlessly headless.
   - Verify that the symbolic logic rules in the agent are cleanly separated from the PyTorch neural components.
4. **Testing**:
   - Ensure new features have accompanying unit tests in the `tests/` directory.

## Workflow
- When asked to review a specific file or feature, read the file carefully, identify potential bugs, suggest performance improvements, and output a clean `diff` or patched code block.
