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
        "scores": torch.zeros((2, 2))
    }
    is_declarer = torch.tensor([[1.0], [0.0]])
    action_mask = torch.ones((2, 54), dtype=torch.bool)
    
    probs, value = agent(obs_dict, is_declarer, mode="normal", action_mask=action_mask)
    
    assert probs.shape == (2, 54)
    assert value.shape == (2, 1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))

def test_ppo_agent_get_action() -> None:
    agent = PPOMultiHeadAgent()
    
    obs_dict = {
        "hand": torch.zeros(32),
        "trick_history": torch.full((30,), -1),
        "deduction_flags": torch.zeros(12),
        "trump_suit": torch.zeros(4),
        "lead_suit": torch.zeros(4),
        "scores": torch.zeros(2)
    }
    is_declarer = torch.tensor([1.0])
    action_mask = torch.ones(54, dtype=torch.bool)
    action_mask[10:] = False # Only first 10 actions valid
    
    action, log_prob, entropy, value = agent.get_action_and_value(
        obs_dict, is_declarer, mode="betli", action_mask=action_mask
    )
    
    assert action.shape == (1,)
    assert 0 <= action.item() < 10
    assert log_prob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1, 1)
