import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, Any
from torch.distributions.categorical import Categorical
from agent.symbolic.masking import masked_softmax

class PPOMultiHeadAgent(nn.Module):
    def __init__(self, hidden_dim: int = 256, action_dim: int = 54) -> None:
        super(PPOMultiHeadAgent, self).__init__()
        
        # 32 cards + 1 padding token (index 32 for -1)
        self.card_embedding = nn.Embedding(33, 16)
        
        # LSTM processes sequence of 30 tricks
        self.lstm = nn.LSTM(input_size=16, hidden_size=64, batch_first=True)
        
        # Flat dims: hand(32) + flags(12) + trump(4) + lead(4) + scores(2) + is_declarer(1) = 55
        # Plus LSTM output(64) = 119
        obs_dim = 119
        
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
        self.policy_ulti = nn.Linear(hidden_dim, action_dim)
        self.policy_talon = nn.Linear(hidden_dim, action_dim)
        self.policy_rob = nn.Linear(hidden_dim, action_dim)
        
    def _preprocess(self, obs_dict: Dict[str, Any], is_declarer: Union[float, int, torch.Tensor]) -> torch.Tensor:
        model_device = next(self.parameters()).device
        hand = torch.as_tensor(obs_dict["hand"], dtype=torch.float32, device=model_device)
        deduction_flags = torch.as_tensor(obs_dict["deduction_flags"], dtype=torch.float32, device=model_device)
        trump_suit = torch.as_tensor(obs_dict["trump_suit"], dtype=torch.float32, device=model_device)
        lead_suit = torch.as_tensor(obs_dict["lead_suit"], dtype=torch.float32, device=model_device)
        scores = torch.as_tensor(obs_dict["scores"], dtype=torch.float32, device=model_device)
        
        # Trick history processing
        history = torch.as_tensor(obs_dict["trick_history"], dtype=torch.long, device=model_device)
        
        # Handle batch dimension consistently
        if hand.dim() == 1:
            hand = hand.unsqueeze(0)
            deduction_flags = deduction_flags.unsqueeze(0)
            trump_suit = trump_suit.unsqueeze(0)
            lead_suit = lead_suit.unsqueeze(0)
            scores = scores.unsqueeze(0)
            history = history.unsqueeze(0)
            
            if not isinstance(is_declarer, torch.Tensor):
                is_declarer = torch.tensor([is_declarer], dtype=torch.float32, device=model_device).unsqueeze(0)
            elif is_declarer.dim() == 1:
                is_declarer = is_declarer.to(model_device).unsqueeze(0)
            else:
                is_declarer = is_declarer.to(model_device)
        else:
            if not isinstance(is_declarer, torch.Tensor):
                is_declarer = torch.tensor(is_declarer, dtype=torch.float32, device=model_device).view(-1, 1)
            elif is_declarer.dim() == 1:
                is_declarer = is_declarer.to(model_device).view(-1, 1)
            else:
                is_declarer = is_declarer.to(model_device)
                
        is_declarer = is_declarer.float()
        
        # LSTM Processing
        # Map -1 to 32 (padding token index)
        history = torch.where(history == -1, torch.tensor(32, device=model_device), history)
        emb_history = self.card_embedding(history) # Shape: (batch, 30, 16)
        lstm_out, _ = self.lstm(emb_history)       # Shape: (batch, 30, 64)
        lstm_final = lstm_out[:, -1, :]            # Take final hidden state
        
        x = torch.cat([
            hand.float(),
            deduction_flags.float(),
            trump_suit.float(),
            lead_suit.float(),
            scores,
            is_declarer,
            lstm_final
        ], dim=-1)
        return x

    def forward(self, obs_dict: Dict[str, torch.Tensor], is_declarer: Union[float, int, torch.Tensor], mode: str = "normal", action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._preprocess(obs_dict, is_declarer)
        features = self.feature_extractor(x)
        value = self.value_head(features)
        
        if mode == "betli":
            logits = self.policy_betli(features)
        elif mode == "durchmars":
            logits = self.policy_durchmars(features)
        elif mode == "ulti":
            logits = self.policy_ulti(features)
        elif mode == "talon":
            logits = self.policy_talon(features)
        elif mode == "decision_to_rob":
            logits = self.policy_rob(features)
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
        
        if action_mask is not None:
            # Re-apply mask to ensure invalid actions stay exactly 0
            probs = probs * action_mask.bool()
            
        probs = probs + 1e-10
        if action_mask is not None:
            probs = probs * action_mask.bool()
            
        probs = probs / probs.sum(dim=-1, keepdim=True)
            
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy(), value
