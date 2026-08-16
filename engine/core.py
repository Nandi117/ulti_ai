import random
from enum import Enum
from typing import List

class Suit(Enum):
    ACORNS = "Acorns"
    LEAVES = "Leaves"
    HEARTS = "Hearts"
    BELLS = "Bells"

class Rank(Enum):
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    UNDER = "Under"
    OVER = "Over"
    KING = "King"
    ACE = "Ace"

class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return f"Card({self.suit.name}, {self.rank.name})"
        
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

class Deck:
    def __init__(self):
        self.cards: List[Card] = [Card(suit, rank) for suit in Suit for rank in Rank]
        
    def shuffle(self):
        random.shuffle(self.cards)
        
    def deal(self, num_cards: int) -> List[Card]:
        if num_cards > len(self.cards):
            raise ValueError("Not enough cards in the deck")
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards
