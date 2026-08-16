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
    bids = []
    # Action 0: Pass
    bids.append(Bid(id=0, name="Pass", points=0))
    
    combinations = []
    
    # 1. Normal Games (Passz, 40-100, 20-100, Ulti)
    for hund in [None, "40-100", "20-100"]:
        for ulti in [False, True]:
            pts = 0
            name_parts = []
            
            if hund == "40-100":
                pts += 4
                name_parts.append("40-100")
            elif hund == "20-100":
                pts += 8
                name_parts.append("20-100")
            
            if ulti:
                name_parts.append("ulti")
            
            # Base values
            if not hund and not ulti:
                pts = 1
                name_parts.append("passz")
            elif not hund and ulti:
                pts = 5 # 4 (ulti) + 1 (passz base)
            elif hund == "40-100" and not ulti:
                pts = 4
            elif hund == "40-100" and ulti:
                pts = 8
            elif hund == "20-100" and not ulti:
                pts = 8
            elif hund == "20-100" and ulti:
                pts = 12
                
            combinations.append({
                "name": " ".join(name_parts),
                "points": pts,
                "has_40_100": hund == "40-100",
                "has_20_100": hund == "20-100",
                "has_ulti": ulti,
                "is_betli": False,
                "is_durchmars": False,
                "is_teritett": False
            })
            
    # 2. Betli Games
    for teritett in [False, True]:
        pts = 10 if teritett else 5
        name = "terített betli" if teritett else "betli"
        combinations.append({
            "name": name,
            "points": pts,
            "has_40_100": False,
            "has_20_100": False,
            "has_ulti": False,
            "is_betli": True,
            "is_durchmars": False,
            "is_teritett": teritett
        })
        
    # 3. Durchmars Games
    for hund in [None, "40-100", "20-100"]:
        for ulti in [False, True]:
            for teritett in [False, True]:
                pts = 0
                name_parts = []
                
                if hund == "40-100":
                    pts += 4
                    name_parts.append("40-100")
                elif hund == "20-100":
                    pts += 8
                    name_parts.append("20-100")
                    
                if ulti:
                    pts += 4
                    name_parts.append("ulti")
                    
                if teritett:
                    pts += 12
                    name_parts.append("terített")
                else:
                    pts += 6
                    
                name_parts.append("durchmars")
                
                combinations.append({
                    "name": " ".join(name_parts),
                    "points": pts,
                    "has_40_100": hund == "40-100",
                    "has_20_100": hund == "20-100",
                    "has_ulti": ulti,
                    "is_betli": False,
                    "is_durchmars": True,
                    "is_teritett": teritett
                })
                
    # Add Piros multiplier
    final_combos = []
    for c in combinations:
        c["is_piros"] = False
        final_combos.append(c)
        c_piros = c.copy()
        c_piros["name"] = "piros " + c["name"]
        c_piros["points"] *= 2
        c_piros["is_piros"] = True
        final_combos.append(c_piros)
        
    # Sort deterministically by points, then alphabetically by name to avoid randomness
    final_combos.sort(key=lambda x: (x["points"], x["name"]))
    
    bid_id = 1
    for c in final_combos:
        bids.append(Bid(
            id=bid_id,
            name=c["name"].capitalize(),
            points=c["points"],
            is_piros=c["is_piros"],
            has_40_100=c["has_40_100"],
            has_20_100=c["has_20_100"],
            has_ulti=c["has_ulti"],
            is_betli=c["is_betli"],
            is_durchmars=c["is_durchmars"],
            is_teritett=c["is_teritett"]
        ))
        bid_id += 1
        
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
