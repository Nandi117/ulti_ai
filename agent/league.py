import random
import copy
import torch
from typing import List, Tuple, Optional
from agent.ppo import PPOMultiHeadAgent


class League:
    """Maintains a pool of frozen past agent snapshots for league training.
    
    During training, opponents are randomly drawn from this pool instead of 
    always using the current agent. This prevents strategy collapse and forces
    the agent to develop robust, generalizable play.
    """
    
    def __init__(self, max_snapshots: int = 10):
        self.snapshots: List[Tuple[dict, dict]] = []  # (declarer_state_dict, defender_state_dict)
        self.max_snapshots = max_snapshots
    
    def add_snapshot(self, declarer_agent: PPOMultiHeadAgent, defender_agent: PPOMultiHeadAgent):
        """Save current weights as a frozen opponent pair."""
        snapshot = (
            {k: v.clone().cpu() for k, v in declarer_agent.state_dict().items()},
            {k: v.clone().cpu() for k, v in defender_agent.state_dict().items()}
        )
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)  # Remove oldest
    
    def sample_opponent(self, device: torch.device) -> Tuple[PPOMultiHeadAgent, PPOMultiHeadAgent]:
        """Return a random frozen opponent pair loaded onto the given device."""
        snapshot = random.choice(self.snapshots)
        opp_decl = PPOMultiHeadAgent().to(device)
        opp_def = PPOMultiHeadAgent().to(device)
        opp_decl.load_state_dict(snapshot[0])
        opp_def.load_state_dict(snapshot[1])
        opp_decl.eval()
        opp_def.eval()
        return opp_decl, opp_def
    
    def has_opponents(self) -> bool:
        """Check if the league has any snapshots available."""
        return len(self.snapshots) > 0
    
    def __len__(self) -> int:
        return len(self.snapshots)
