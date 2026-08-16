import numpy as np
from typing import List, Optional, Union
from engine.core import Card, Suit, Rank

RANK_POWER_STANDARD = {
    Rank.SEVEN: 1,
    Rank.EIGHT: 2,
    Rank.NINE: 3,
    Rank.UNDER: 4,
    Rank.OVER: 5,
    Rank.KING: 6,
    Rank.TEN: 7,
    Rank.ACE: 8
}

RANK_POWER_BETLI = {
    Rank.SEVEN: 1,
    Rank.EIGHT: 2,
    Rank.NINE: 3,
    Rank.TEN: 4,
    Rank.UNDER: 5,
    Rank.OVER: 6,
    Rank.KING: 7,
    Rank.ACE: 8
}

# Generate all cards in the same order as Deck
ALL_CARDS = [Card(suit, rank) for suit in Suit for rank in Rank]

class Trick:
    def __init__(self, trump_suit: Optional[Suit] = None, is_betli_or_durchmars: bool = False):
        self.trump_suit = trump_suit
        self.is_betli_or_durchmars = is_betli_or_durchmars
        self.cards_played: List[Card] = []
        self.players: List[int] = [] # To keep track of who played which card
        self.lead_suit: Optional[Suit] = None
        self.rank_power = RANK_POWER_BETLI if is_betli_or_durchmars else RANK_POWER_STANDARD

    def play_card(self, player_id: int, card: Card):
        if not self.cards_played:
            self.lead_suit = card.suit
        self.cards_played.append(card)
        self.players.append(player_id)

    def get_winner(self) -> Optional[int]:
        if not self.cards_played:
            return None
            
        best_card = self.cards_played[0]
        best_player = self.players[0]
        
        for card, player in zip(self.cards_played[1:], self.players[1:]):
            if self.trump_suit is not None and card.suit == self.trump_suit:
                if best_card.suit != self.trump_suit:
                    best_card = card
                    best_player = player
                elif self.rank_power[card.rank] > self.rank_power[best_card.rank]:
                    best_card = card
                    best_player = player
            elif card.suit == self.lead_suit and best_card.suit != self.trump_suit:
                if self.rank_power[card.rank] > self.rank_power[best_card.rank]:
                    best_card = card
                    best_player = player
                    
        return best_player

def get_action_mask(hand: Union[List[int], np.ndarray], trick: Trick) -> np.ndarray:
    """
    Returns a boolean array of length 32 indicating which cards are legal to play.
    hand: either a list of card IDs (0-31) or a boolean array of length 32.
    """
    mask = np.zeros(32, dtype=bool)
    if isinstance(hand, np.ndarray) and len(hand) == 32:
        hand_ids = np.where(hand > 0)[0]
    else:
        hand_ids = hand
        
    hand_ids = list(hand_ids) # Ensure it's a list

    if not trick.cards_played:
        # Lead can be anything in hand
        for c_id in hand_ids:
            mask[c_id] = True
        return mask

    lead_suit = trick.lead_suit
    
    # 1. Check if we have cards of the lead suit
    lead_suit_cards = [c_id for c_id in hand_ids if ALL_CARDS[c_id].suit == lead_suit]
    
    if lead_suit_cards:
        # Must follow suit, and overtrick if possible
        highest_lead_rank_power = 0
        for card in trick.cards_played:
            if card.suit == lead_suit:
                highest_lead_rank_power = max(highest_lead_rank_power, trick.rank_power[card.rank])
                
        overtrick_cards = [c_id for c_id in lead_suit_cards if trick.rank_power[ALL_CARDS[c_id].rank] > highest_lead_rank_power]
        
        if overtrick_cards:
            for c_id in overtrick_cards:
                mask[c_id] = True
        else:
            for c_id in lead_suit_cards:
                mask[c_id] = True
        return mask

    # 2. Cannot follow suit. Must play trump if we have one.
    if trick.trump_suit is not None:
        trump_cards = [c_id for c_id in hand_ids if ALL_CARDS[c_id].suit == trick.trump_suit]
        if trump_cards:
            # Overtrick trump if possible
            highest_trump_power = 0
            for card in trick.cards_played:
                if card.suit == trick.trump_suit:
                    highest_trump_power = max(highest_trump_power, trick.rank_power[card.rank])
            
            overtrick_trumps = [c_id for c_id in trump_cards if trick.rank_power[ALL_CARDS[c_id].rank] > highest_trump_power]
            
            if overtrick_trumps:
                for c_id in overtrick_trumps:
                    mask[c_id] = True
            else:
                for c_id in trump_cards:
                    mask[c_id] = True
            return mask

    # 3. Cannot follow suit and have no trumps (or no trump in game)
    for c_id in hand_ids:
        mask[c_id] = True
        
    return mask
