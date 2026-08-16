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
        (1, 2, "Piros passz", True, False, False, False, False, False, False),
        (4, 8, "40-100", False, True, False, False, False, False, False),
        (5, 5, "Négy ász", False, False, False, False, False, False, True),
        (5, 9, "Ulti", False, False, False, True, False, False, False),
        (5, 5, "Betli", False, False, False, False, True, False, False),
        (8, 12, "Durchmars", False, False, False, False, False, True, False),
        (8, 12, "Színtelen durchmars", False, False, False, False, False, True, False),
        (8, 16, "40-100 négy ász", False, True, False, False, False, False, True),
        (8, 16, "40-100 ulti", False, True, False, True, False, False, False),
        (8, 16, "Piros 40-100", True, True, False, False, False, False, False),
        (8, 16, "20-100", False, False, True, False, False, False, False),
        (9, 17, "Ulti négy ász", False, False, False, True, False, False, True),
        (10, 10, "Piros négy ász", True, False, False, False, False, False, True),
        (10, 18, "Piros ulti", True, False, False, True, False, False, False),
        (10, 10, "Piros betli vagy rebetli", True, False, False, False, True, False, False),
        (10, 20, "40-100 durchmars", False, True, False, False, False, True, False),
        (10, 20, "Ulti durchmars", False, False, False, True, False, True, False),
        (12, 24, "40-100 ulti négy ász", False, True, False, True, False, False, True),
        (12, 24, "20-100 négy ász", False, False, True, False, False, False, True),
        (12, 24, "20-100 ulti", False, False, True, True, False, False, False),
        (12, 24, "Redurchmars", False, False, False, False, False, True, False),
        (12, 24, "Piros durchmars", True, False, False, False, False, True, False),
        (12, 24, "Terített durchmars", False, False, False, False, False, True, True),
        (14, 28, "40-100 ulti durchmars", False, True, False, True, False, True, False),
        (14, 28, "20-100 durchmars", False, False, True, False, False, True, False),
        (16, 32, "20-100 ulti négy ász", False, False, True, True, False, False, True),
        (16, 32, "Piros 40-100 ulti", True, True, False, True, False, False, False),
        (16, 32, "Piros 40-100 négy ász", True, True, False, False, False, False, True),
        (16, 32, "Piros 20-100", True, False, True, False, False, False, False),
        (16, 32, "40-100 terített durchmars", False, True, False, False, False, True, True),
        (16, 32, "Ulti terített durchmars", False, False, False, True, False, True, True),
        (18, 34, "Piros ulti négy ász", True, False, False, True, False, False, True),
        (18, 36, "20-100 ulti durchmars", False, False, True, True, False, True, False),
        (20, 40, "40-100 ulti terített durchmars", False, True, False, True, False, True, True),
        (20, 40, "Piros 40-100 durchmars", True, True, False, False, False, True, False),
        (20, 40, "Piros ulti durchmars", True, False, False, True, False, True, False),
        (20, 40, "20-100 terített durchmars", False, False, True, False, False, True, True),
        (20, 20, "Terített betli", False, False, False, False, True, False, True),
        (24, 48, "20-100 ulti terített durchmars", False, False, True, True, False, True, True),
        (24, 48, "Piros 40-100 ulti négy ász", True, True, False, True, False, False, True),
        (24, 48, "Piros 20-100 négy ász", True, False, True, False, False, False, True),
        (24, 48, "Piros 20-100 ulti", True, False, True, True, False, False, False),
        (24, 48, "Piros terített durchmars", True, False, False, False, False, True, True),
        (24, 48, "Színtelen terített durchmars", False, False, False, False, False, True, True),
        (28, 56, "Piros 40-100 ulti durchmars", True, True, False, True, False, True, False),
        (28, 56, "Piros 20-100 durchmars", True, False, True, False, False, True, False),
        (32, 64, "Piros 20-100 ulti négy ász", True, False, True, True, False, False, True),
        (32, 64, "Piros 40-100 terített durchmars", True, True, False, False, False, True, True),
        (32, 64, "Piros ulti terített durchmars", True, False, False, True, False, True, True),
        (36, 72, "Piros 20-100 ulti durchmars", True, False, True, True, False, True, False),
        (40, 80, "Piros 40-100 ulti terített durchmars", True, True, False, True, False, True, True),
        (40, 80, "Piros 20-100 terített durchmars", True, False, True, False, False, True, True),
        (48, 96, "Piros 20-100 ulti terített durchmars", True, False, True, True, False, True, True)
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
        if not mask[bid_id]:
            raise ValueError(f"Illegal bid {bid_id}")
            
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
