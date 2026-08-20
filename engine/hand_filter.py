from typing import List, Dict
from engine.core import Card, Suit, Rank
from engine.bidding import ALL_BIDS

def evaluate_hand_for_bids(hand_cards: List[Card]) -> Dict[str, bool]:
    allowed = {
        "passz": True,
        "40-100": True,
        "piros 40-100": True,
        "ulti": True,
        "piros ulti": True,
        "betli": True,
        "piros betli": True,
        "durchmars": True,
        "piros durchmars": True
    }
    
    # 40-100 logic: MATHEMATICAL IMPOSSIBILITY
    # You cannot possibly score 100 points without at least one marriage (20 or 40 pts)
    has_hearts_marriage = any(c.suit == Suit.HEARTS and c.rank == Rank.KING for c in hand_cards) and \
                          any(c.suit == Suit.HEARTS and c.rank == Rank.OVER for c in hand_cards)
    
    suits_with_marriage = []
    for suit in Suit:
        if any(c.suit == suit and c.rank == Rank.KING for c in hand_cards) and \
           any(c.suit == suit and c.rank == Rank.OVER for c in hand_cards):
            suits_with_marriage.append(suit)
            
    if len(suits_with_marriage) == 0:
        allowed["40-100"] = False
        allowed["piros 40-100"] = False
    elif not has_hearts_marriage:
        allowed["piros 40-100"] = False
        
    # NOTE: We previously had heuristic filters here that blocked Betli with Aces,
    # Durchmars without 3 Aces, and Ulti without 7+3 trumps.
    # These have been removed to let the neural network explore and learn strategically!
    # The exceptional hand filter (which forced Passz off) was also removed because
    # Reward Shaping (bid_bonus) solves the exact same problem much more elegantly.

    return allowed

def apply_hand_filter_to_mask(hand_cards: List[Card], raw_mask: List[bool]) -> List[bool]:
    allowed_bids = evaluate_hand_for_bids(hand_cards)
    filtered_mask = list(raw_mask)
    
    for i, is_legal in enumerate(raw_mask):
        if not is_legal or i == 0:
            continue
            
        if i < len(ALL_BIDS):
            bid = ALL_BIDS[i]
            # Exact match (e.g. "piros ulti", "ulti")
            bid_name = bid.name.split(" (")[0].lower()
            if bid_name in allowed_bids and not allowed_bids[bid_name]:
                filtered_mask[i] = False
                
    return filtered_mask
