from typing import List, Dict, Set
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
    
    aces = [c for c in hand_cards if c.rank == Rank.ACE]
    kings = [c for c in hand_cards if c.rank == Rank.KING]
    overs = [c for c in hand_cards if c.rank == Rank.OVER]
    sevens = [c for c in hand_cards if c.rank == Rank.SEVEN]
    eights = [c for c in hand_cards if c.rank == Rank.EIGHT]
    nines = [c for c in hand_cards if c.rank == Rank.NINE]
    
    # 40-100 logic
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
        
    # Betli logic
    high_cards_count = len(aces) + len(kings) + len(overs)
    if high_cards_count > 2 or len(aces) > 0:
        allowed["betli"] = False
        allowed["piros betli"] = False
        
    if high_cards_count == 0 and (len(sevens) + len(eights) + len(nines)) >= 6:
        allowed["passz"] = False
        allowed["durchmars"] = False
        allowed["piros durchmars"] = False
        
    # Durchmars logic
    tens = [c for c in hand_cards if c.rank == Rank.TEN]
    high_cards_total = len(aces) + len(tens) + len(kings) + len(overs)
    
    # Durchmars requires winning every trick. Without at least 3 Aces and massive high card density, it is impossible.
    if len(aces) < 3 or high_cards_total < 7:
        allowed["durchmars"] = False
        allowed["piros durchmars"] = False
        
    if high_cards_total >= 8 and len(aces) >= 3:
        allowed["passz"] = False
        allowed["betli"] = False
        allowed["piros betli"] = False
        
    # Ulti logic
    can_play_ulti = False
    can_play_piros_ulti = False
    has_strong_ulti = False
    
    for suit in Suit:
        suit_cards = [c for c in hand_cards if c.suit == suit]
        has_vii = any(c.rank == Rank.SEVEN for c in suit_cards)
        has_ace = any(c.rank == Rank.ACE for c in suit_cards)
        
        if has_vii and len(suit_cards) >= 3:
            can_play_ulti = True
            if suit == Suit.HEARTS:
                can_play_piros_ulti = True
                
        if has_vii and has_ace and len(suit_cards) >= 4:
            has_strong_ulti = True
            
    if not can_play_ulti:
        allowed["ulti"] = False
    if not can_play_piros_ulti:
        allowed["piros ulti"] = False
        
    if has_strong_ulti:
        allowed["passz"] = False
        
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
                
    if not allowed_bids["passz"]:
        if any(filtered_mask[1:41]):
            filtered_mask[0] = False
        else:
            filtered_mask[0] = True
            
    return filtered_mask
