import pytest
import numpy as np
from engine.bidding import ALL_BIDS, BIDS_BY_ID, Auction

def test_bids_generation():
    assert len(ALL_BIDS) > 0
    # ID 0 should be Pass
    assert ALL_BIDS[0].id == 0
    assert ALL_BIDS[0].name == "Pass"
    assert ALL_BIDS[0].points == 0
    
    # Check max bid points (48 according to rules: Piros 20-100 ulti teritett durchmars)
    max_bid = max(ALL_BIDS, key=lambda b: b.points)
    assert max_bid.points == 48
    assert max_bid.is_piros is True
    assert max_bid.has_20_100 is True
    assert max_bid.has_ulti is True
    assert max_bid.is_teritett is True
    assert max_bid.is_durchmars is True
    
    # Ensure IDs are continuous
    ids = [b.id for b in ALL_BIDS]
    assert ids == list(range(len(ALL_BIDS)))

def test_auction_initial_mask():
    auction = Auction()
    mask = auction.get_action_mask()
    
    # All bids (except Pass) have >0 points, so all should be valid initially
    assert np.all(mask) == True
    assert mask[0] == True # Can pass
    
def test_auction_step_and_mask_update():
    auction = Auction(starting_player=0)
    
    # Find a mid-tier bid, e.g., points == 10
    bid_10 = next(b for b in ALL_BIDS if b.points == 10)
    auction.step(bid_10.id)
    
    assert auction.highest_bid.id == bid_10.id
    assert auction.highest_bidder == 0
    assert auction.active_player == 1
    assert auction.passes_in_a_row == 0
    assert not auction.is_over
    
    mask = auction.get_action_mask()
    # Pass should be valid
    assert mask[0] == True
    # Bids with <= 10 points should be invalid
    for b in ALL_BIDS:
        if b.id != 0:
            if b.points <= 10:
                assert not mask[b.id]
            else:
                assert mask[b.id]

def test_auction_end_with_three_passes():
    auction = Auction(starting_player=0)
    auction.step(0) # P0 passes
    assert auction.active_player == 1
    assert not auction.is_over
    
    auction.step(0) # P1 passes
    assert auction.active_player == 2
    assert not auction.is_over
    
    auction.step(0) # P2 passes
    assert auction.is_over
    assert auction.highest_bid is None
    
    # Mask should be all False when over
    mask = auction.get_action_mask()
    assert not np.any(mask)

def test_auction_end_after_bid():
    auction = Auction(starting_player=0)
    
    bid_4 = next(b for b in ALL_BIDS if b.points == 4)
    auction.step(bid_4.id) # P0 bids 4 points
    assert auction.highest_bid.id == bid_4.id
    
    auction.step(0) # P1 passes
    assert not auction.is_over
    assert auction.passes_in_a_row == 1
    
    auction.step(0) # P2 passes
    assert auction.is_over
    assert auction.passes_in_a_row == 2
    assert auction.highest_bidder == 0

def test_auction_overbidding():
    auction = Auction(starting_player=0)
    
    bid_4 = next(b for b in ALL_BIDS if b.points == 4)
    bid_6 = next(b for b in ALL_BIDS if b.points == 6)
    
    auction.step(bid_4.id) # P0 bids 4
    auction.step(0)        # P1 passes
    auction.step(bid_6.id) # P2 bids 6 (valid overbid)
    
    assert auction.highest_bid.id == bid_6.id
    assert auction.highest_bidder == 2
    assert auction.passes_in_a_row == 0
    assert not auction.is_over
    
    auction.step(0) # P0 passes
    auction.step(0) # P1 passes
    assert auction.is_over
    assert auction.highest_bidder == 2

def test_invalid_bid_raises():
    auction = Auction(starting_player=0)
    bid_10 = next(b for b in ALL_BIDS if b.points == 10)
    bid_5 = next(b for b in ALL_BIDS if b.points == 5)
    
    auction.step(bid_10.id)
    
    with pytest.raises(ValueError, match="Illegal bid"):
        auction.step(bid_5.id) # Cannot bid lower
        
    with pytest.raises(ValueError, match="Illegal bid"):
        auction.step(bid_10.id) # Cannot bid same value

def test_step_after_over_raises():
    auction = Auction(starting_player=0)
    auction.step(0)
    auction.step(0)
    auction.step(0)
    assert auction.is_over
    
    with pytest.raises(ValueError, match="Auction is already over"):
        auction.step(0)
