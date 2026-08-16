import torch
import numpy as np
from agent.symbolic.masking import masked_softmax
from agent.baselines.heuristic import HeuristicAgent

def test_masked_softmax():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    
    probs = masked_softmax(logits, mask)
    
    # Check probabilities of masked actions are 0.0
    assert probs[0, 1].item() == 0.0
    assert probs[1, 0].item() == 0.0
    assert probs[1, 2].item() == 0.0
    
    # Check probabilities sum to 1.0 along the specified dim
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
    
    # Check non-masked probabilities are positive
    assert probs[0, 0].item() > 0.0
    assert probs[0, 2].item() > 0.0
    assert probs[1, 1].item() == 1.0

def test_heuristic_agent():
    agent = HeuristicAgent()
    mask = np.array([True, False, True, False])
    
    for _ in range(10):
        action = agent.act(None, mask)
        assert action in [0, 2]
