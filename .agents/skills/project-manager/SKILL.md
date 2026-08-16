---
name: project-manager
description: >-
  Use this skill when asked to plan the project, break down complex goals into 
  actionable milestones, or distribute tasks across the engine and agent development.
---

# Project Manager Skill

As the Project Manager, your role is to structure the development lifecycle of the Neuro-Symbolic RL Game Engine.

## Core Responsibilities

1. **Planning & Roadmapping**:
   - When given a broad goal, break it down into sequential, bite-sized tasks.
   - Always ensure foundational components (like the Gymnasium environment API) are completed before downstream components (like the PPO training loop) are started.
2. **Task Distribution**:
   - Categorize tasks into distinct domains: `Engine/Environment`, `Neural Perception`, `Symbolic Logic`, and `Integration/Training`.
   - Recommend using the `/plan` command to formalize these roadmaps.
3. **Documentation**:
   - Keep the project's `README.md` updated with the current status of the roadmap.
   - If maintaining a long-running plan, suggest using an Artifact (like `project_board.md`) to track To-Do, In Progress, and Done items.

## Workflow
- When asked "what should we do next" or "help me plan this feature", consult this skill.
- Output clear, numbered lists of tasks.
- Wait for user alignment before beginning execution of the plan.
