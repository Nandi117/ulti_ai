import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from engine.core import Deck, Suit, Rank, Card

def test_deck_initialization():
    deck = Deck()
    assert len(deck.cards) == 32
    
    expected_cards = [Card(suit, rank) for suit in Suit for rank in Rank]
    for card in expected_cards:
        assert card in deck.cards

def test_deck_shuffle():
    deck1 = Deck()
    deck2 = Deck()
    deck2.shuffle()
    
    assert deck1.cards != deck2.cards
    assert len(deck2.cards) == 32
    # Also verify all cards are still there
    for card in deck1.cards:
        assert card in deck2.cards

def test_deck_deal():
    deck = Deck()
    dealt = deck.deal(5)
    
    assert len(dealt) == 5
    assert len(deck.cards) == 27
    
    dealt2 = deck.deal(27)
    assert len(dealt2) == 27
    assert len(deck.cards) == 0

def test_deck_deal_too_many():
    deck = Deck()
    with pytest.raises(ValueError, match="Not enough cards in the deck"):
        deck.deal(33)
