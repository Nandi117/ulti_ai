import pytest
import torch
import numpy as np
from agent.ppo import PPOMultiHeadAgent

def test_ppo_agent_forward() -> None:
    agent = PPOMultiHeadAgent()

    obs_dict = {
        "hand": torch.zeros((2, 32)),
        "trick_history": torch.full((2, 30), -1),
        "deduction_flags": torch.zeros((2, 12)),
        "trump_suit": torch.zeros((2, 4)),
        "lead_suit": torch.zeros((2, 4)),
        "scores": torch.zeros((2, 2)),
        "belief_state": torch.zeros((2, 4, 32)),
        "public_belief_state": torch.zeros((2, 4, 32)),
        "talon_first_drop": torch.tensor([[32], [32]])
    }
    action_mask = torch.ones((2, 54), dtype=torch.bool)

    probs, value = agent(obs_dict, mode="normal", action_mask=action_mask)

    assert probs.shape == (2, 54)
    assert value.shape == (2, 1)
    
    # Check mask applied properly - shouldn't contain NaNs
    assert not torch.isnan(probs).any()
    # Masked probabilities should sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))

def test_ppo_agent_get_action() -> None:
    agent = PPOMultiHeadAgent()

    obs_dict = {
        "hand": torch.zeros(32),
        "trick_history": torch.full((30,), -1),
        "deduction_flags": torch.zeros(12),
        "trump_suit": torch.zeros(4),
        "lead_suit": torch.zeros(4),
        "scores": torch.zeros(2),
        "belief_state": torch.zeros((4, 32)),
        "public_belief_state": torch.zeros((4, 32)),
        "talon_first_drop": torch.tensor([32])
    }
    action_mask = torch.ones(54, dtype=torch.bool)
    action_mask[10:] = False # Only first 10 actions valid

    action, log_prob, entropy, value = agent.get_action_and_value(
        obs_dict, mode="betli", action_mask=action_mask
    )

    assert action.shape == (1,)
    assert action.item() < 10 # Should respect mask
    assert log_prob.shape == (1,)
    assert value.shape == (1, 1)
    assert value.shape == (1, 1)
