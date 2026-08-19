from typing import List, Dict, Set
from engine.core import Card, Suit, Rank
from engine.bidding import ALL_BIDS

def evaluate_hand_for_bids(hand_cards: List[Card], force_high_game: bool = False) -> Dict[str, bool]:
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
    if high_cards_count > 3 or len(aces) > 0:
        allowed["betli"] = False
        allowed["piros betli"] = False
        
    if high_cards_count == 0 and (len(sevens) + len(eights) + len(nines)) >= 6:
        allowed["durchmars"] = False
        allowed["piros durchmars"] = False
        
    # Durchmars logic
    tens = [c for c in hand_cards if c.rank == Rank.TEN]
    high_cards_total = len(aces) + len(tens) + len(kings) + len(overs)
    suits_in_hand = set(c.suit for c in hand_cards)
    
    can_durchmars = False
    if len(aces) >= 3 and high_cards_total >= 9:
        can_durchmars = True
    elif len(aces) == 2 and len(suits_in_hand) <= 2 and high_cards_total >= 10:
        can_durchmars = True
        
    if not can_durchmars:
        allowed["durchmars"] = False
        allowed["piros durchmars"] = False
        
    if high_cards_total >= 8 and (len(aces) >= 3 or (len(aces) == 2 and len(suits_in_hand) <= 2)):
        allowed["betli"] = False
        allowed["piros betli"] = False
        
    # Ulti logic
    can_play_ulti = False
    can_play_piros_ulti = False
    
    exceptional_ulti = False
    exceptional_piros_ulti = False
    
    for suit in Suit:
        suit_cards = [c for c in hand_cards if c.suit == suit]
        has_vii = any(c.rank == Rank.SEVEN for c in suit_cards)
        has_ace = any(c.rank == Rank.ACE for c in suit_cards)
        
        if has_vii and len(suit_cards) >= 3:
            if suit == Suit.HEARTS:
                can_play_piros_ulti = True
            else:
                can_play_ulti = True
                
        # Exceptional Ulti: 7, Ace, and at least 5 trumps total
        if has_vii and has_ace and len(suit_cards) >= 5:
            if suit == Suit.HEARTS:
                exceptional_piros_ulti = True
            else:
                exceptional_ulti = True
                
    if not can_play_ulti:
        allowed["ulti"] = False
    if not can_play_piros_ulti:
        allowed["piros ulti"] = False
        
    # Exceptional Checks
    exceptional_betli = (high_cards_count == 0) # Only 7s, 8s, 9s, 10s, Unders
    exceptional_durchmars = (len(aces) == 4 and len(kings) >= 2)
    exceptional_40_100 = len(suits_with_marriage) >= 1 and len(aces) >= 3
    
    if force_high_game:
        # We only force passing OFF if the hand is mathematically exceptional (80-95% win rate)
        # This guarantees the agent gets positive reinforcement for bidding high games.
        if exceptional_betli or exceptional_durchmars or exceptional_ulti or exceptional_piros_ulti or exceptional_40_100:
            allowed["passz"] = False
            allowed["piros passz"] = False
            
            # Optionally, we can also force them into the SPECIFIC exceptional game
            # to make learning even faster, but just banning passz is enough since
            # the action mask will still allow the good bids.

    return allowed

def apply_hand_filter_to_mask(hand_cards: List[Card], raw_mask: List[bool], force_high_game: bool = False) -> List[bool]:
    allowed_bids = evaluate_hand_for_bids(hand_cards, force_high_game=force_high_game)
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
