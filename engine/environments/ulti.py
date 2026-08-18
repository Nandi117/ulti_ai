import gymnasium as gym
from gymnasium import spaces
import numpy as np

from engine.core import Deck, Suit, Rank
from engine.bidding import Auction, ALL_BIDS
from engine.trick import Trick, get_action_mask, ALL_CARDS

class UltiEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, curriculum_mode: bool = False, curriculum_level: int = 1, training_filter_mode: bool = False):
        super().__init__()
        
        self.curriculum_mode = curriculum_mode
        self.curriculum_level = curriculum_level
        self.training_filter_mode = training_filter_mode
        
        # Bidding uses up to 41 actions, playing uses 0-31
        self.action_space = spaces.Discrete(54)
        
        self.observation_space = spaces.Dict({
            "hand": spaces.MultiBinary(32),
            "trick_history": spaces.Box(low=-1, high=31, shape=(30,), dtype=np.int8),
            "deduction_flags": spaces.MultiBinary(12), # 3 players * 4 suits void
            "trump_suit": spaces.MultiBinary(4),
            "lead_suit": spaces.MultiBinary(4),
            "scores": spaces.Box(low=0, high=120, shape=(2,), dtype=np.float32)
        })
        
        # Setup attributes
        self.hands = []
        self.talon = []
        self.current_player = 0
        self.phase = "bidding"
        self.auction = None
        self.trick = None
        self.history = np.full(30, -1, dtype=np.int8)
        self.history_idx = 0
        self.deduction_flags = np.zeros(12, dtype=np.int8)
        self.trump_suit = None
        self.tricks_played = 0
        self.declarer_tricks_won = 0
        self.defenders_tricks_won = 0
        self.declarer_points = 0
        self.defenders_points = 0
        self.declarer_had_4_aces = False

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        
        deck = Deck()
        
        if self.curriculum_mode:
            import random
            from engine.bidding import ALL_BIDS
            
            deck.shuffle()
            
            active_level = self.curriculum_level
            if active_level == "mixed":
                active_level = random.choice([1, 4, 5, 6])
                
            if active_level in [3, 4]:
                # Bids: "Ulti"
                valid_bids = [b for b in ALL_BIDS if b.name == "Ulti"]
            elif active_level == 5:
                # Bids: "Betli"
                valid_bids = [b for b in ALL_BIDS if b.name == "Betli"]
            elif active_level == 6:
                # Bids: "Durchmars"
                valid_bids = [b for b in ALL_BIDS if b.name == "Durchmars"]
            else:
                # LEVEL 1: Only basic point-gathering games
                valid_bids = [b for b in ALL_BIDS if b.id in [0, 1]]
                
            forced_bid = random.choice(valid_bids)
            declarer_id = random.randint(0, 2)
            
            if forced_bid.is_piros:
                self.trump_suit = Suit.HEARTS
            else:
                if active_level in [5, 6]:
                    self.trump_suit = None
                else:
                    self.trump_suit = random.choice([Suit.ACORNS, Suit.BELLS, Suit.LEAVES])
                
            declarer_cards = []
            if active_level in [1, 3, 4]:
                # LEVEL 1 & 3 & 4: Give the Declarer trumps to guarantee a strong hand
                all_trumps = [c for c in deck.cards if c.suit == self.trump_suit]
                
                if active_level in [3, 4]:
                    # In Level 3 and 4 (Ulti), explicitly give the 7 of trumps
                    seven_trump = next(c for c in all_trumps if c.rank == Rank.SEVEN)
                    declarer_cards.append(seven_trump)
                    deck.cards.remove(seven_trump)
                    all_trumps.remove(seven_trump)
                    
                if active_level == 4:
                    # In Level 4, explicitly give the Ace and 10 of trumps
                    required_ranks = [Rank.ACE, Rank.TEN]
                    for rank in required_ranks:
                        trump_card = next(c for c in all_trumps if c.rank == rank)
                        declarer_cards.append(trump_card)
                        deck.cards.remove(trump_card)
                        all_trumps.remove(trump_card)
                    
                random.shuffle(all_trumps)
                target_trumps = 5 if active_level in [3, 4] else 4
                for c in all_trumps[:target_trumps - len(declarer_cards)]:
                    declarer_cards.append(c)
                    deck.cards.remove(c)
                    
            if active_level == 5:
                # LEVEL 5 (Betli Puzzle): Give the Declarer very low cards
                low_ranks = [Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN]
                all_low_cards = [c for c in deck.cards if c.rank in low_ranks]
                random.shuffle(all_low_cards)
                
                # Deal 8 low cards to the Declarer
                for c in all_low_cards[:8]:
                    declarer_cards.append(c)
                    deck.cards.remove(c)
                    
            if active_level == 6:
                # LEVEL 6 (Durchmars Puzzle): Give the Declarer very high cards (all Aces and Kings)
                high_ranks = [Rank.ACE, Rank.KING]
                all_high_cards = [c for c in deck.cards if c.rank in high_ranks]
                random.shuffle(all_high_cards)
                
                # Deal 8 high cards to the Declarer
                for c in all_high_cards[:8]:
                    declarer_cards.append(c)
                    deck.cards.remove(c)
                
            # Deal the rest of the random cards (LEVEL 2 deals 10 completely random cards here)
            deck.shuffle()
            while len(declarer_cards) < 10:
                c = deck.deal(1)[0]
                declarer_cards.append(c)
                
            self.talon = deck.deal(2)
            def1_cards = deck.deal(10)
            def2_cards = deck.deal(10)
            
            # Put hands in order
            hands_cards = [[], [], []]
            hands_cards[declarer_id] = declarer_cards
            hands_cards[(declarer_id + 1) % 3] = def1_cards
            hands_cards[(declarer_id + 2) % 3] = def2_cards
            
            self.hands = [self._cards_to_binary(hc) for hc in hands_cards]
            
            self.phase = "playing"
            self.auction = Auction(starting_player=0)
            self.auction.highest_bid = forced_bid
            self.auction.highest_bidder = declarer_id
            self.auction.is_over = True
                
            self.trick = Trick(trump_suit=self.trump_suit, is_betli_or_durchmars=False)
            self.current_player = declarer_id
            
            # Set declarer_had_4_aces and compute initial points
            aces_count = sum(1 for c in declarer_cards if c.rank == Rank.ACE)
            self.declarer_had_4_aces = (aces_count == 4)
            self.declarer_points = 0
            self.defenders_points = 0
            
            for i in range(3):
                player_hand_ids = np.where(self.hands[i])[0]
                player_cards = [ALL_CARDS[c_id] for c_id in player_hand_ids]
                for suit in Suit:
                    has_king = any(c.suit == suit and c.rank == Rank.KING for c in player_cards)
                    has_over = any(c.suit == suit and c.rank == Rank.OVER for c in player_cards)
                    if has_king and has_over:
                        pts = 40 if suit == self.trump_suit else 20
                        if i == self.auction.highest_bidder:
                            self.declarer_points += pts
                        else:
                            self.defenders_points += pts
                            
        else:
            deck.shuffle()
            self.hands = [
                self._cards_to_binary(deck.deal(10)),
                self._cards_to_binary(deck.deal(10)),
                self._cards_to_binary(deck.deal(10))
            ]
            self.talon = deck.deal(2)
            self.current_player = 0
            self.phase = "bidding"
            self.auction = Auction(starting_player=self.current_player)
            self.trick = Trick()
            self.trump_suit = None
            self.declarer_points = 0
            self.defenders_points = 0
            self.declarer_had_4_aces = False
            
        self.history = np.full(30, -1, dtype=np.int8)
        self.history_idx = 0
        self.deduction_flags = np.zeros(12, dtype=np.int8)
        self.tricks_played = 0
        self.declarer_tricks_won = 0
        self.defenders_tricks_won = 0
        
        return self._get_obs(), self._get_info()

    def _cards_to_binary(self, cards: list) -> np.ndarray:
        bin_hand = np.zeros(32, dtype=np.int8)
        for c in cards:
            idx = ALL_CARDS.index(c)
            bin_hand[idx] = 1
        return bin_hand

    def _get_obs(self) -> dict:
        suit_to_idx = {Suit.ACORNS: 0, Suit.BELLS: 1, Suit.LEAVES: 2, Suit.HEARTS: 3}
        
        trump_arr = np.zeros(4, dtype=np.int8)
        if self.trump_suit is not None:
            trump_arr[suit_to_idx[self.trump_suit]] = 1
            
        lead_arr = np.zeros(4, dtype=np.int8)
        if self.trick is not None and self.trick.lead_suit is not None:
            lead_arr[suit_to_idx[self.trick.lead_suit]] = 1
            
        return {
            "hand": self.hands[self.current_player].copy(),
            "trick_history": self.history.copy(),
            "deduction_flags": self.deduction_flags.copy(),
            "trump_suit": trump_arr,
            "lead_suit": lead_arr,
            "scores": np.array([self.declarer_points, self.defenders_points], dtype=np.float32)
        }

    def _get_info(self) -> dict:
        mask = np.zeros(54, dtype=np.int8)
        if self.phase == "bidding":
            bidding_mask = self.auction.get_action_mask()
            mask[:len(bidding_mask)] = bidding_mask
        else:
            playing_mask = get_action_mask(self.hands[self.current_player], self.trick)
            mask[:32] = playing_mask
            
        return {"action_mask": mask}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        action = int(action)
        reward = 0.0
        terminated = False
        truncated = False

        if self.phase == "bidding":
            mask = self.auction.get_action_mask()
            if action >= len(mask) or not mask[action]:
                raise ValueError(f"Invalid bid action {action}")
                
            self.auction.step(action)
            
            if self.auction.is_over:
                self.phase = "playing"
                if self.auction.highest_bid is None:
                    terminated = True
                    return self._get_obs(), reward, terminated, truncated, self._get_info()
                
                # Setup trick phase
                # Setup trick phase based on the Bid
                bid = self.auction.highest_bid
                if bid.is_piros:
                    self.trump_suit = Suit.HEARTS
                elif bid.is_betli:
                    self.trump_suit = None
                else:
                    # Auto-pick the suit the declarer has the most of (including Hearts, as requested by user)
                    declarer_hand = self.hands[self.auction.highest_bidder]
                    suit_counts = {Suit.ACORNS: 0, Suit.LEAVES: 0, Suit.BELLS: 0, Suit.HEARTS: 0}
                    for c_id in np.where(declarer_hand)[0]:
                        suit = ALL_CARDS[c_id].suit
                        if suit in suit_counts:
                            suit_counts[suit] += 1
                    self.trump_suit = max(suit_counts, key=suit_counts.get)
                    
                self.trick = Trick(trump_suit=self.trump_suit, is_betli_or_durchmars=bid.is_betli or bid.is_durchmars)
                self.current_player = self.auction.highest_bidder
                
                # Check 4 Aces and Marriages
                declarer_hand_ids = np.where(self.hands[self.auction.highest_bidder])[0]
                declarer_cards = [ALL_CARDS[c_id] for c_id in declarer_hand_ids]
                aces_count = sum(1 for c in declarer_cards if c.rank == Rank.ACE)
                self.declarer_had_4_aces = (aces_count == 4)
                
                for i in range(3):
                    player_hand_ids = np.where(self.hands[i])[0]
                    player_cards = [ALL_CARDS[c_id] for c_id in player_hand_ids]
                    for suit in Suit:
                        has_king = any(c.suit == suit and c.rank == Rank.KING for c in player_cards)
                        has_over = any(c.suit == suit and c.rank == Rank.OVER for c in player_cards)
                        if has_king and has_over:
                            pts = 40 if suit == self.trump_suit else 20
                            if i == self.auction.highest_bidder:
                                self.declarer_points += pts
                            else:
                                self.defenders_points += pts
            else:
                self.current_player = self.auction.active_player
                
        elif self.phase == "playing":
            if action >= 32:
                raise ValueError(f"Invalid play action {action}")
                
            mask = get_action_mask(self.hands[self.current_player], self.trick)
            if not mask[action]:
                raise ValueError(f"Invalid play action {action}")
                
            card = ALL_CARDS[action]
            self.hands[self.current_player][action] = 0
            
            self.trick.play_card(self.current_player, card)
            self.history[self.history_idx] = action
            self.history_idx += 1
            
            # Neuro-symbolic deduction flags update
            if self.trick.lead_suit and card.suit != self.trick.lead_suit:
                suit_idx = list(Suit).index(self.trick.lead_suit)
                flag_idx = self.current_player * 4 + suit_idx
                self.deduction_flags[flag_idx] = 1
                
            if len(self.trick.cards_played) == 3:
                winner = self.trick.get_winner()
                
                bid = self.auction.highest_bid
                if not bid.is_betli and (not bid.is_durchmars or bid.has_40_100 or bid.has_20_100):
                    trick_points = sum(10 for c in self.trick.cards_played if c.rank in [Rank.TEN, Rank.ACE])
                    if winner == self.auction.highest_bidder:
                        self.declarer_tricks_won += 1
                        self.declarer_points += trick_points
                    else:
                        self.defenders_tricks_won += 1
                        self.defenders_points += trick_points
                else:
                    if winner == self.auction.highest_bidder:
                        self.declarer_tricks_won += 1
                    else:
                        self.defenders_tricks_won += 1
                
                self.tricks_played += 1
                
                # EARLY TERMINATION CHECKS
                failed_early = False
                if bid.is_betli and self.declarer_tricks_won > 0:
                    failed_early = True
                if bid.is_durchmars and self.defenders_tricks_won > 0:
                    failed_early = True
                if bid.has_ulti and self.tricks_played < 10:
                    for card in self.trick.cards_played:
                        if card.suit == self.trump_suit and card.rank == Rank.SEVEN:
                            failed_early = True
                            
                if failed_early:
                    terminated = True
                    reward = float(-bid.points)
                elif self.tricks_played == 10:
                    terminated = True
                    
                    if not bid.is_betli and (not bid.is_durchmars or bid.has_40_100 or bid.has_20_100):
                        if winner == self.auction.highest_bidder:
                            self.declarer_points += 10
                        else:
                            self.defenders_points += 10
                            
                    declarer_won = True
                    
                    if not bid.is_betli and not bid.is_durchmars and not bid.has_40_100 and not bid.has_20_100:
                        if self.declarer_points <= self.defenders_points:
                            declarer_won = False
                            
                    if bid.has_40_100 or bid.has_20_100:
                        if self.declarer_points < 100:
                            declarer_won = False
                            
                    if bid.has_ulti:
                        winning_card_index = self.trick.players.index(winner)
                        winning_card = self.trick.cards_played[winning_card_index]
                        if not (winner == self.auction.highest_bidder and winning_card.suit == self.trump_suit and winning_card.rank == Rank.SEVEN):
                            declarer_won = False
                            
                    if "négy ász" in bid.name.lower():
                        if not self.declarer_had_4_aces or self.declarer_points <= self.defenders_points:
                            declarer_won = False
                            
                    if bid.is_betli:
                        if self.declarer_tricks_won > 0:
                            declarer_won = False
                    if bid.is_durchmars:
                        if self.declarer_tricks_won < 10:
                            declarer_won = False
                            
                    # Double the reward for winning to encourage risk-taking in bidding phase
                    reward = float(bid.points * 2) if declarer_won else float(-bid.points)
                
                self.current_player = winner
                if not terminated:
                    self.trick = Trick(trump_suit=self.trump_suit, is_betli_or_durchmars=bid.is_betli or bid.is_durchmars)
            else:
                self.current_player = (self.current_player + 1) % 3

        return self._get_obs(), reward, terminated, truncated, self._get_info()
