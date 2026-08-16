import pytest
import numpy as np
from engine.core import Card, Suit, Rank
from engine.trick import Trick, get_action_mask, ALL_CARDS, RANK_POWER

def get_card_id(suit: Suit, rank: Rank) -> int:
    for i, c in enumerate(ALL_CARDS):
        if c.suit == suit and c.rank == rank:
            return i
    raise ValueError("Card not found")

def test_trick_winner_no_trump():
    trick = Trick()
    # Acorns 9 (power 3)
    c1 = Card(Suit.ACORNS, Rank.NINE)
    # Acorns King (power 6)
    c2 = Card(Suit.ACORNS, Rank.KING)
    # Acorns 10 (power 7)
    c3 = Card(Suit.ACORNS, Rank.TEN)
    # Hearts Ace (power 8, but not lead suit and no trump)
    c4 = Card(Suit.HEARTS, Rank.ACE)
    
    trick.play_card(0, c1)
    trick.play_card(1, c2)
    trick.play_card(2, c3)
    trick.play_card(3, c4)
    
    assert trick.get_winner() == 2

def test_trick_winner_with_trump():
    trick = Trick(trump_suit=Suit.HEARTS)
    # Acorns 10 (power 7)
    c1 = Card(Suit.ACORNS, Rank.TEN)
    # Acorns Ace (power 8)
    c2 = Card(Suit.ACORNS, Rank.ACE)
    # Hearts 7 (power 1, trump!)
    c3 = Card(Suit.HEARTS, Rank.SEVEN)
    # Hearts 9 (power 3, higher trump!)
    c4 = Card(Suit.HEARTS, Rank.NINE)
    
    trick.play_card(0, c1)
    trick.play_card(1, c2)
    trick.play_card(2, c3)
    trick.play_card(3, c4)
    
    assert trick.get_winner() == 3

def test_action_mask_lead():
    trick = Trick(trump_suit=Suit.HEARTS)
    hand = [
        get_card_id(Suit.ACORNS, Rank.SEVEN),
        get_card_id(Suit.HEARTS, Rank.ACE)
    ]
    mask = get_action_mask(hand, trick)
    
    assert mask[get_card_id(Suit.ACORNS, Rank.SEVEN)] == True
    assert mask[get_card_id(Suit.HEARTS, Rank.ACE)] == True
    assert mask.sum() == 2

def test_action_mask_follow_suit_overtrick():
    trick = Trick(trump_suit=Suit.HEARTS)
    trick.play_card(0, Card(Suit.ACORNS, Rank.OVER)) # power 5
    
    hand = [
        get_card_id(Suit.ACORNS, Rank.SEVEN), # power 1
        get_card_id(Suit.ACORNS, Rank.KING),  # power 6
        get_card_id(Suit.HEARTS, Rank.ACE)    # trump, power 8
    ]
    mask = get_action_mask(hand, trick)
    
    # Must follow suit and overtrick -> only Acorns King is legal
    assert mask[get_card_id(Suit.ACORNS, Rank.KING)] == True
    assert mask[get_card_id(Suit.ACORNS, Rank.SEVEN)] == False
    assert mask[get_card_id(Suit.HEARTS, Rank.ACE)] == False
    assert mask.sum() == 1

def test_action_mask_follow_suit_no_overtrick():
    trick = Trick(trump_suit=Suit.HEARTS)
    trick.play_card(0, Card(Suit.ACORNS, Rank.TEN)) # power 7
    
    hand = [
        get_card_id(Suit.ACORNS, Rank.SEVEN), # power 1
        get_card_id(Suit.ACORNS, Rank.KING),  # power 6
        get_card_id(Suit.HEARTS, Rank.ACE)    # trump, power 8
    ]
    mask = get_action_mask(hand, trick)
    
    # Must follow suit, but can't overtrick -> all acorns are legal
    assert mask[get_card_id(Suit.ACORNS, Rank.KING)] == True
    assert mask[get_card_id(Suit.ACORNS, Rank.SEVEN)] == True
    assert mask[get_card_id(Suit.HEARTS, Rank.ACE)] == False
    assert mask.sum() == 2

def test_action_mask_cannot_follow_suit_must_trump():
    trick = Trick(trump_suit=Suit.HEARTS)
    trick.play_card(0, Card(Suit.ACORNS, Rank.TEN)) # power 7
    
    hand = [
        get_card_id(Suit.BELLS, Rank.SEVEN), # power 1
        get_card_id(Suit.HEARTS, Rank.SEVEN), # power 1 (trump)
        get_card_id(Suit.HEARTS, Rank.KING)  # power 6 (trump)
    ]
    mask = get_action_mask(hand, trick)
    
    assert mask[get_card_id(Suit.BELLS, Rank.SEVEN)] == False
    assert mask[get_card_id(Suit.HEARTS, Rank.SEVEN)] == True
    assert mask[get_card_id(Suit.HEARTS, Rank.KING)] == True
    assert mask.sum() == 2

def test_action_mask_must_overtrick_trump():
    trick = Trick(trump_suit=Suit.HEARTS)
    trick.play_card(0, Card(Suit.ACORNS, Rank.TEN))
    trick.play_card(1, Card(Suit.HEARTS, Rank.NINE)) # trump played by someone else
    
    hand = [
        get_card_id(Suit.BELLS, Rank.SEVEN), 
        get_card_id(Suit.HEARTS, Rank.SEVEN), # lower trump
        get_card_id(Suit.HEARTS, Rank.KING)   # higher trump
    ]
    mask = get_action_mask(hand, trick)
    
    # Must play higher trump
    assert mask[get_card_id(Suit.HEARTS, Rank.KING)] == True
    assert mask[get_card_id(Suit.HEARTS, Rank.SEVEN)] == False
    assert mask[get_card_id(Suit.BELLS, Rank.SEVEN)] == False
    assert mask.sum() == 1

def test_action_mask_cannot_follow_no_trump():
    trick = Trick(trump_suit=Suit.HEARTS)
    trick.play_card(0, Card(Suit.ACORNS, Rank.TEN))
    
    hand = [
        get_card_id(Suit.BELLS, Rank.SEVEN), 
        get_card_id(Suit.LEAVES, Rank.KING)
    ]
    mask = get_action_mask(hand, trick)
    
    # No lead suit, no trump -> can play anything
    assert mask[get_card_id(Suit.BELLS, Rank.SEVEN)] == True
    assert mask[get_card_id(Suit.LEAVES, Rank.KING)] == True
    assert mask.sum() == 2

def test_action_mask_boolean_array():
    trick = Trick(trump_suit=Suit.HEARTS)
    trick.play_card(0, Card(Suit.ACORNS, Rank.OVER))
    
    hand_ids = [
        get_card_id(Suit.ACORNS, Rank.SEVEN),
        get_card_id(Suit.ACORNS, Rank.KING),
        get_card_id(Suit.HEARTS, Rank.ACE)
    ]
    hand_arr = np.zeros(32, dtype=bool)
    hand_arr[hand_ids] = True
    
    mask = get_action_mask(hand_arr, trick)
    assert mask[get_card_id(Suit.ACORNS, Rank.KING)] == True
    assert mask[get_card_id(Suit.ACORNS, Rank.SEVEN)] == False
    assert mask[get_card_id(Suit.HEARTS, Rank.ACE)] == False
    assert mask.sum() == 1

def test_action_mask_follow_suit_with_trump_played():
    trick = Trick(trump_suit=Suit.HEARTS)
    trick.play_card(0, Card(Suit.ACORNS, Rank.TEN))
    trick.play_card(1, Card(Suit.HEARTS, Rank.SEVEN)) # Someone trumped!
    
    hand = [
        get_card_id(Suit.ACORNS, Rank.NINE), # power 3
        get_card_id(Suit.ACORNS, Rank.ACE),  # power 8
        get_card_id(Suit.BELLS, Rank.SEVEN)
    ]
    mask = get_action_mask(hand, trick)
    
    # Must follow suit AND overtrick the highest ACORN, even though a trump is played!
    assert mask[get_card_id(Suit.ACORNS, Rank.ACE)] == True
    assert mask[get_card_id(Suit.ACORNS, Rank.NINE)] == False
    assert mask[get_card_id(Suit.BELLS, Rank.SEVEN)] == False
    assert mask.sum() == 1
