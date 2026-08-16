import numpy as np
from typing import Any, Optional

class HeuristicAgent:
    """
    A simple baseline agent that selects a random legal move based on the provided mask.
    """
    def __init__(self, action_space: Optional[Any] = None) -> None:
        self.action_space = action_space

    def act(self, observation: Any, mask: np.ndarray) -> int:
        """
        Selects a random legal action.
        
        Args:
            observation: The environment observation.
            mask: A boolean numpy array indicating valid actions (True = valid).
            
        Returns:
            An integer representing the selected action.
        """
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available according to the mask.")
        return np.random.choice(valid_actions)
