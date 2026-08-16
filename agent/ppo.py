import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, Any
from torch.distributions.categorical import Categorical
from agent.symbolic.masking import masked_softmax

class PPOMultiHeadAgent(nn.Module):
    def __init__(self, obs_dim: int = 75, hidden_dim: int = 256, action_dim: int = 41) -> None:
        super(PPOMultiHeadAgent, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self.policy_normal = nn.Linear(hidden_dim, action_dim)
        self.policy_betli = nn.Linear(hidden_dim, action_dim)
        self.policy_durchmars = nn.Linear(hidden_dim, action_dim)
        
    def _preprocess(self, obs_dict: Dict[str, torch.Tensor], is_declarer: Union[float, int, torch.Tensor]) -> torch.Tensor:
        hand = obs_dict["hand"].float()
        trick_history = obs_dict["trick_history"].float()
        deduction_flags = obs_dict["deduction_flags"].float()
        
        if hand.dim() == 1:
            hand = hand.unsqueeze(0)
            trick_history = trick_history.unsqueeze(0)
            deduction_flags = deduction_flags.unsqueeze(0)
            if not isinstance(is_declarer, torch.Tensor):
                is_declarer = torch.tensor([is_declarer], dtype=torch.float32).unsqueeze(0)
            elif is_declarer.dim() == 1:
                is_declarer = is_declarer.unsqueeze(0)
        else:
            if not isinstance(is_declarer, torch.Tensor):
                is_declarer = torch.tensor(is_declarer, dtype=torch.float32).view(-1, 1)
            elif is_declarer.dim() == 1:
                is_declarer = is_declarer.view(-1, 1)
                
        is_declarer = is_declarer.float()
        x = torch.cat([hand, trick_history, deduction_flags, is_declarer], dim=-1)
        return x

    def forward(self, obs_dict: Dict[str, torch.Tensor], is_declarer: Union[float, int, torch.Tensor], mode: str = "normal", action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._preprocess(obs_dict, is_declarer)
        features = self.feature_extractor(x)
        value = self.value_head(features)
        
        if mode == "betli":
            logits = self.policy_betli(features)
        elif mode == "durchmars":
            logits = self.policy_durchmars(features)
        else:
            logits = self.policy_normal(features)
            
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            probs = masked_softmax(logits, action_mask.bool(), dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            
        return probs, value
        
    def get_action_and_value(self, obs_dict: Dict[str, torch.Tensor], is_declarer: Union[float, int, torch.Tensor], mode: str = "normal", action_mask: Optional[torch.Tensor] = None, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, value = self.forward(obs_dict, is_declarer, mode, action_mask)
        
        # Add small epsilon to avoid zero probabilities for Categorical
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)
            
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy(), value
