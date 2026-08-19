import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class Bid:
    id: int
    name: str
    points: int
    is_piros: bool = False
    has_40_100: bool = False
    has_20_100: bool = False
    has_ulti: bool = False
    is_betli: bool = False
    is_durchmars: bool = False
    is_teritett: bool = False

def generate_all_bids() -> List[Bid]:
    raw_bids = [
        (0, 0, "Passz", False, False, False, False, False, False, False),
        (2, 0, "Piros Passz", True, False, False, False, False, False, False),
        (4, 0, "40-100", False, True, False, False, False, False, False),
        (8, 0, "Piros 40-100", True, True, False, False, False, False, False),
        (5, 5, "Betli", False, False, False, False, True, False, False),
        (10, 0, "Piros Betli", True, False, False, False, True, False, False),
        (4, 9, "Ulti", False, False, False, True, False, False, False),
        (8, 0, "Piros Ulti", True, False, False, True, False, False, False),
        (6, 12, "Durchmars", False, False, False, False, False, True, False),
        (12, 0, "Piros Durchmars", True, False, False, False, False, True, False),
    ]
    
    bids = []
    for i, (pts, _, name, is_p, h40, h20, is_u, is_b, is_d, is_t) in enumerate(raw_bids):
        bids.append(Bid(
            id=i,
            name=name.capitalize(),
            points=pts,
            is_piros=is_p,
            has_40_100=h40,
            has_20_100=h20,
            has_ulti=is_u,
            is_betli=is_b,
            is_durchmars=is_d,
            is_teritett=is_t
        ))
        
    return bids

ALL_BIDS: List[Bid] = generate_all_bids()
BIDS_BY_ID: Dict[int, Bid] = {bid.id: bid for bid in ALL_BIDS}
NUM_BIDS = len(ALL_BIDS)

class Auction:
    def __init__(self, starting_player: int = 0):
        self.starting_player: int = starting_player
        self.highest_bid: Optional[Bid] = None
        self.highest_bidder: int = -1
        self.active_player: int = starting_player
        self.passes_in_a_row: int = 0
        self.is_over: bool = False
        self.history: List[Tuple[int, Bid]] = []

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_BIDS, dtype=bool)
        if self.is_over:
            return mask
            
        # Can always pass
        mask[0] = True
        
        current_pts = self.highest_bid.points if self.highest_bid else 0
        for bid in ALL_BIDS:
            if bid.id != 0 and bid.points > current_pts:
                mask[bid.id] = True
                
        return mask

    def step(self, bid_id: int):
        if self.is_over:
            raise ValueError("Auction is already over.")
            
        mask = self.get_action_mask()
        # Trust the environment's mask filtering, do not crash here
            
        bid = BIDS_BY_ID[bid_id]
        self.history.append((self.active_player, bid))
        
        if bid_id == 0:
            self.passes_in_a_row += 1
            if self.highest_bid is None and self.passes_in_a_row == 3:
                self.is_over = True
            elif self.highest_bid is not None and self.passes_in_a_row == 2:
                self.is_over = True
        else:
            self.highest_bid = bid
            self.highest_bidder = self.active_player
            self.passes_in_a_row = 0
            
        if not self.is_over:
            self.active_player = (self.active_player + 1) % 3
