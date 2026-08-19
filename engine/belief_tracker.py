import numpy as np
from typing import List, Tuple
from engine.core import Card, Suit, Rank
from engine.trick import ALL_CARDS

class BeliefTracker:
    def __init__(self):
        # Viewers: 0, 1, 2
        # Targets: 0, 1, 2, 3 (Talon)
        # Cards: 32
        self.constraints = np.ones((3, 4, 32), dtype=bool)
        self.definite = np.zeros((3, 4, 32), dtype=bool)
        self.capacities = np.array([0, 0, 0, 0], dtype=np.int8)
        
        # Public belief: what any observer knows from public actions only
        # Shape: (4, 32) — constraints on where each card could be
        self.public_constraints = np.ones((4, 32), dtype=bool)
        self.public_played = np.zeros(32, dtype=bool)  # Cards that have been played publicly
        
    def reset_deal(self, hands: List[np.ndarray], starting_player: int):
        self.constraints.fill(True)
        self.definite.fill(False)
        
        # Reset public belief
        self.public_constraints.fill(True)
        self.public_played.fill(False)
        
        sizes = [10, 10, 10]
        sizes[starting_player] = 12
        self.capacities = np.array([sizes[0], sizes[1], sizes[2], 0], dtype=np.int8)
        
        # Each player knows their own hand
        for viewer in range(3):
            for c in range(32):
                if hands[viewer][c] == 1:
                    self.definite[viewer, viewer, c] = True
                    # If viewer has it, no one else can have it (handled by logic, but let's strictly constrain it)
                    for t in range(4):
                        if t != viewer:
                            self.constraints[viewer, t, c] = False

    def drop_talon(self, player_id: int, card_indices: List[int]):
        """Player drops cards into the talon."""
        self.capacities[player_id] -= len(card_indices)
        self.capacities[3] += len(card_indices)
        
        for viewer in range(3):
            if viewer == player_id:
                # The player who drops definitely knows these cards are in the talon
                for c in card_indices:
                    self.definite[viewer, viewer, c] = False
                    self.definite[viewer, 3, c] = True
            else:
                # For other viewers, any card that player_id COULD have, COULD now be in the talon!
                for c in range(32):
                    if self.definite[viewer, player_id, c]:
                        self.definite[viewer, player_id, c] = False
                        self.constraints[viewer, player_id, c] = True
                        self.constraints[viewer, 3, c] = True
                    elif self.constraints[viewer, player_id, c]:
                        self.constraints[viewer, 3, c] = True
            
    def rob_talon(self, player_id: int, original_talon_cards: List[int]):
        """Player picks up the talon."""
        self.capacities[player_id] += self.capacities[3]
        self.capacities[3] = 0
        
        # The player who robs it now knows they have these cards.
        for c in original_talon_cards:
            self.definite[player_id, 3, c] = False
            self.definite[player_id, player_id, c] = True
            for t in range(4):
                if t != player_id:
                    self.constraints[player_id, t, c] = False
            
        # For any viewer who definitely knew a card was in the talon, they now know player_id has it!
        for viewer in range(3):
            if viewer == player_id:
                continue
            for c in range(32):
                if self.definite[viewer, 3, c]:
                    self.definite[viewer, 3, c] = False
                    self.definite[viewer, player_id, c] = True
                
                # Any card that COULD have been in the talon, COULD now be in player_id's hand!
                # Even if player_id was previously void in that suit, they just picked it up!
                if self.constraints[viewer, 3, c]:
                    self.constraints[viewer, player_id, c] = True
                    self.constraints[viewer, 3, c] = False

    def mark_void(self, player_id: int, suit: Suit):
        """Public knowledge: player_id has no cards of this suit."""
        for c in range(32):
            if ALL_CARDS[c].suit == suit:
                self.constraints[:, player_id, c] = False
                self.public_constraints[player_id, c] = False  # Public knowledge too
                
    def mark_cannot_overtrick(self, player_id: int, suit: Suit, max_rank_value: int):
        """Public knowledge: player_id cannot beat the given rank value in this suit."""
        for c in range(32):
            if ALL_CARDS[c].suit == suit and ALL_CARDS[c].rank.value > max_rank_value:
                self.constraints[:, player_id, c] = False
                self.public_constraints[player_id, c] = False  # Public knowledge too
                
    def mark_public_knowledge(self, player_id: int, card_idx: int):
        """A player announces 40/20, revealing they hold this card."""
        for viewer in range(3):
            self.definite[viewer, player_id, card_idx] = True
            for t in range(4):
                if t != player_id:
                    self.constraints[viewer, t, card_idx] = False
                    
    def play_card(self, player_id: int, card_idx: int):
        """A player plays a card to the trick. The card leaves the tracking system."""
        self.capacities[player_id] -= 1
        for viewer in range(3):
            for t in range(4):
                self.definite[viewer, t, card_idx] = False
                self.constraints[viewer, t, card_idx] = False
        # Public: card is now played (visible to all), remove from all locations
        self.public_played[card_idx] = True
        for t in range(4):
            self.public_constraints[t, card_idx] = False

    def get_probabilities(self, viewer_id: int) -> np.ndarray:
        """Returns 4x32 matrix of probabilities from the perspective of viewer_id."""
        probs = np.zeros((4, 32), dtype=np.float32)
        
        eff_cap = self.capacities.copy()
        for t in range(4):
            eff_cap[t] -= np.sum(self.definite[viewer_id, t])
            
        for c in range(32):
            # 1. Definite knowledge
            found = False
            for t in range(4):
                if self.definite[viewer_id, t, c]:
                    probs[t, c] = 1.0
                    found = True
                    break
            if found:
                continue
                
            # 2. Distribute by effective capacity and constraints
            valid_targets = []
            total_weight = 0
            for t in range(4):
                if self.constraints[viewer_id, t, c] and eff_cap[t] > 0:
                    valid_targets.append(t)
                    total_weight += eff_cap[t]
                    
            if total_weight > 0:
                for t in valid_targets:
                    probs[t, c] = eff_cap[t] / total_weight
                    
        return probs

    def get_public_probabilities(self, target_player: int) -> np.ndarray:
        """Returns 4x32 matrix: what ANY observer would believe about all players' hands
        based solely on public information (cards played, voids revealed, marriages announced).
        
        This is the 'Public Belief State' that enables bluffing — the agent can see
        what the opponents think about its hand based on its public actions."""
        probs = np.zeros((4, 32), dtype=np.float32)
        
        # Use current capacities for weighting
        eff_cap = self.capacities.copy().astype(np.float32)
        
        for c in range(32):
            # Skip played cards
            if self.public_played[c]:
                continue
                
            # Distribute based on public constraints and capacity
            valid_targets = []
            total_weight = 0.0
            for t in range(4):
                if self.public_constraints[t, c] and eff_cap[t] > 0:
                    valid_targets.append(t)
                    total_weight += eff_cap[t]
                    
            if total_weight > 0:
                for t in valid_targets:
                    probs[t, c] = eff_cap[t] / total_weight
                    
        return probs

