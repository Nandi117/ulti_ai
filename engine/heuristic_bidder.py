from typing import List
from engine.core import Card, Suit, Rank

class OracleBidder:
    @staticmethod
    def get_best_bid(declarer_hand: List[Card], d1_hand: List[Card], d2_hand: List[Card]) -> int:
        if OracleBidder._can_win_durchmars(declarer_hand, d1_hand, d2_hand):
            if OracleBidder._is_piros_strong(declarer_hand):
                return 9
            return 8
        if OracleBidder._can_win_betli(declarer_hand, d1_hand, d2_hand):
            # We want it to play both Betli and Piros Betli, but if we just hardcode it, it will only play one.
            # To fix the "only plays Piros Betli" bug, we look at the Hearts in hand.
            if OracleBidder._can_win_betli_piros(declarer_hand):
                return 5
            return 4
        best_ulti_suit = OracleBidder._get_ulti_suit(declarer_hand, d1_hand, d2_hand)
        if best_ulti_suit:
            return 7 if best_ulti_suit == Suit.HEARTS else 6
        best_40_100_suit = OracleBidder._get_40_100_suit(declarer_hand, d1_hand, d2_hand)
        if best_40_100_suit:
            return 3 if best_40_100_suit == Suit.HEARTS else 2
        return 1 if OracleBidder._is_piros_strong(declarer_hand) else 0
        
    @staticmethod
    def _can_win_betli(decl: List[Card], d1: List[Card], d2: List[Card]) -> bool:
        has_ace = any(c.rank == Rank.ACE for c in decl)
        if has_ace:
            return False
        has_king = any(c.rank == Rank.KING for c in decl)
        if has_king:
            return False
        return True
        
    @staticmethod
    def _can_win_betli_piros(decl: List[Card]) -> bool:
        # Only play Piros Betli if the hand is heavily stacked with Hearts 
        # (or lack thereof) to naturally split Betli and Piros Betli outcomes.
        hearts = sum(1 for c in decl if c.suit == Suit.HEARTS)
        return hearts == 0

    @staticmethod
    def _can_win_durchmars(decl: List[Card], d1: List[Card], d2: List[Card]) -> bool:
        aces = sum(1 for c in decl if c.rank == Rank.ACE)
        tens = sum(1 for c in decl if c.rank == Rank.TEN)
        kings = sum(1 for c in decl if c.rank == Rank.KING)
        if aces == 4 and (tens + kings) >= 3:
            return True
        return False

    @staticmethod
    def _is_piros_strong(decl: List[Card]) -> bool:
        hearts = [c for c in decl if c.suit == Suit.HEARTS]
        return len(hearts) >= 4 and any(c.rank in [Rank.ACE, Rank.TEN] for c in hearts)

    @staticmethod
    def _get_ulti_suit(decl: List[Card], d1: List[Card], d2: List[Card]) -> Suit:
        for suit in Suit:
            trumps = [c for c in decl if c.suit == suit]
            has_vii = any(c.rank == Rank.SEVEN for c in trumps)
            has_high_trump = any(c.rank in [Rank.ACE, Rank.TEN, Rank.KING] for c in trumps)
            if has_vii and has_high_trump and len(trumps) >= 4:
                d1_trumps = sum(1 for c in d1 if c.suit == suit)
                d2_trumps = sum(1 for c in d2 if c.suit == suit)
                if d1_trumps <= 2 and d2_trumps <= 2:
                    return suit
        return None

    @staticmethod
    def _get_40_100_suit(decl: List[Card], d1: List[Card], d2: List[Card]) -> Suit:
        for suit in Suit:
            trumps = [c for c in decl if c.suit == suit]
            has_king = any(c.rank == Rank.KING for c in trumps)
            has_over = any(c.rank == Rank.OVER for c in trumps)
            if has_king and has_over:
                aces = sum(1 for c in decl if c.rank == Rank.ACE)
                tens = sum(1 for c in decl if c.rank == Rank.TEN)
                if (aces + tens) >= 4 and len(trumps) >= 4:
                    return suit
        return None
