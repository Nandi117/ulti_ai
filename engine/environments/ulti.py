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
            "scores": spaces.Box(low=0, high=120, shape=(2,), dtype=np.float32),
            "belief_state": spaces.Box(low=0.0, high=1.0, shape=(4, 32), dtype=np.float32)
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
                self._cards_to_binary(deck.deal(12)),
                self._cards_to_binary(deck.deal(10)),
                self._cards_to_binary(deck.deal(10))
            ]
            self.talon = []
            self.current_player = 0
            self.phase = "drop_talon"
            self.cards_to_drop = 2
            self.must_bid_higher = False
            self.has_robbed = [True, False, False]
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
        
        from engine.belief_tracker import BeliefTracker
        self.belief_tracker = BeliefTracker()
        self.belief_tracker.reset_deal(self.hands, starting_player=0)
        
        # 30% chance to force the agent into a non-pass game if one is legally available
        self.force_high_game = np.random.random() < 0.30
        
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
            
        if hasattr(self, 'belief_tracker'):
            belief_state = self.belief_tracker.get_probabilities(self.current_player)
        else:
            belief_state = np.zeros((4, 32), dtype=np.float32)

        return {
            "hand": self.hands[self.current_player].copy(),
            "trick_history": self.history.copy(),
            "deduction_flags": self.deduction_flags.copy(),
            "trump_suit": trump_arr,
            "lead_suit": lead_arr,
            "scores": np.array([self.declarer_points, self.defenders_points], dtype=np.float32),
            "belief_state": belief_state
        }

    def _get_info(self) -> dict:
        mask = np.zeros(54, dtype=np.int8)
        if self.phase == "drop_talon":
            from engine.trick import ALL_CARDS
            from engine.core import Rank
            for i, has_card in enumerate(self.hands[self.current_player]):
                if has_card:
                    mask[i] = 1
        elif self.phase == "bidding":
            bidding_mask = self.auction.get_action_mask()
            
            if getattr(self, 'training_filter_mode', False):
                from engine.hand_filter import apply_hand_filter_to_mask
                from engine.trick import ALL_CARDS
                
                player_hand_ids = np.where(self.hands[self.current_player])[0]
                player_cards = [ALL_CARDS[c_id] for c_id in player_hand_ids]
                filtered_list = apply_hand_filter_to_mask(player_cards, bidding_mask, force_high_game=getattr(self, 'force_high_game', False))
                bidding_mask = np.array(filtered_list, dtype=np.int8)
                
            if getattr(self, 'must_bid_higher', False):
                bidding_mask[0] = 0 # No Passz after robbing
                if np.sum(bidding_mask) == 0:
                    current_pts = self.auction.highest_bid.points if self.auction.highest_bid else 0
                    from engine.bidding import ALL_BIDS
                    for bid in ALL_BIDS:
                        if bid.points > current_pts:
                            bidding_mask[bid.id] = 1
                            break
                    if np.sum(bidding_mask) == 0:
                        bidding_mask[ALL_BIDS[-1].id] = 1 # Force Piros Durchmars if literally impossible to go higher
                            
            mask[:len(bidding_mask)] = bidding_mask
            
            if not getattr(self, 'must_bid_higher', False) and not self.has_robbed[self.current_player]:
                mask[45] = 1 # Rob Talon action
                
        elif self.phase == "pick_trump":
            bid = self.auction.highest_bid
            from engine.trick import ALL_CARDS
            from engine.core import Rank, Suit
            player_hand_ids = np.where(self.hands[self.current_player])[0]
            player_cards = [ALL_CARDS[c_id] for c_id in player_hand_ids]
            
            if bid and bid.has_ulti:
                if any(c.rank == Rank.SEVEN and c.suit == Suit.ACORNS for c in player_cards): mask[40] = 1
                if any(c.rank == Rank.SEVEN and c.suit == Suit.LEAVES for c in player_cards): mask[41] = 1
                if any(c.rank == Rank.SEVEN and c.suit == Suit.BELLS for c in player_cards): mask[42] = 1
                if np.sum(mask[40:43]) == 0:
                    mask[40] = 1 # Fallback if forced to bluff
            elif bid and bid.has_40_100:
                has_acorns_m = any(c.rank == Rank.KING and c.suit == Suit.ACORNS for c in player_cards) and any(c.rank == Rank.OVER and c.suit == Suit.ACORNS for c in player_cards)
                has_leaves_m = any(c.rank == Rank.KING and c.suit == Suit.LEAVES for c in player_cards) and any(c.rank == Rank.OVER and c.suit == Suit.LEAVES for c in player_cards)
                has_bells_m = any(c.rank == Rank.KING and c.suit == Suit.BELLS for c in player_cards) and any(c.rank == Rank.OVER and c.suit == Suit.BELLS for c in player_cards)
                
                if has_acorns_m: mask[40] = 1
                if has_leaves_m: mask[41] = 1
                if has_bells_m: mask[42] = 1
                if np.sum(mask[40:43]) == 0:
                    mask[40] = 1 # Fallback if forced to bluff
            else:
                mask[40] = 1 # Acorns
                mask[41] = 1 # Leaves
                mask[42] = 1 # Bells
                mask[42] = 1 # Bells
        else:
            playing_mask = get_action_mask(self.hands[self.current_player], self.trick)
            
            # Heuristic: Never play Trump VII in an Ulti game unless it's the only choice or last card
            bid = self.auction.highest_bid
            if bid and bid.has_ulti and self.current_player == self.auction.highest_bidder:
                cards_in_hand = np.sum(self.hands[self.current_player])
                if cards_in_hand > 1:
                    from engine.trick import ALL_CARDS
                    from engine.core import Rank
                    trump_vii_idx = -1
                    for idx, card in enumerate(ALL_CARDS):
                        if card.suit == self.trump_suit and card.rank == Rank.SEVEN:
                            trump_vii_idx = idx
                            break
                    if trump_vii_idx != -1 and playing_mask[trump_vii_idx]:
                        playing_mask[trump_vii_idx] = 0
                        if np.sum(playing_mask) == 0:
                            playing_mask[trump_vii_idx] = 1 # Revert if forced to play it
                            
            mask[:32] = playing_mask
            
        return {"action_mask": mask}


    def _setup_playing_phase(self):
        self.phase = "playing"
        bid = self.auction.highest_bid
        self.trick = Trick(trump_suit=self.trump_suit, is_betli_or_durchmars=bid.is_betli or bid.is_durchmars)
        self.current_player = self.auction.highest_bidder
        
        # Check 4 Aces and Marriages
        from engine.trick import ALL_CARDS
        from engine.core import Rank, Suit
        import numpy as np
        
        declarer_hand_ids = np.where(self.hands[self.auction.highest_bidder])[0]
        declarer_cards = [ALL_CARDS[c_id] for c_id in declarer_hand_ids]
        aces_count = sum(1 for c in declarer_cards if c.rank == Rank.ACE)
        self.declarer_had_4_aces = (aces_count == 4)
        
        for i in range(3):
            player_hand_ids = np.where(self.hands[i])[0]
            player_cards = [ALL_CARDS[c_id] for c_id in player_hand_ids]
            for suit in Suit:
                king_card = next((c for c in player_cards if c.suit == suit and c.rank == Rank.KING), None)
                over_card = next((c for c in player_cards if c.suit == suit and c.rank == Rank.OVER), None)
                if king_card and over_card:
                    pts = 40 if suit == self.trump_suit else 20
                    if i == self.auction.highest_bidder:
                        self.declarer_points += pts
                    else:
                        self.defenders_points += pts
                    
                    if hasattr(self, 'belief_tracker'):
                        self.belief_tracker.mark_public_knowledge(i, ALL_CARDS.index(king_card))
                        self.belief_tracker.mark_public_knowledge(i, ALL_CARDS.index(over_card))

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        from engine.trick import ALL_CARDS
        action = int(action)
        reward = 0.0
        terminated = False
        truncated = False
        
        info = self._get_info()
        mask = info["action_mask"]
        if not mask[action]:
            raise ValueError(f"Invalid action {action} for phase {self.phase}")

        if self.phase == "drop_talon":
            from engine.trick import ALL_CARDS
            card = ALL_CARDS[action]
            self.hands[self.current_player][action] = 0
            self.talon.append(card)
            self.cards_to_drop -= 1
            if self.cards_to_drop == 0:
                self.phase = "bidding"
                if hasattr(self, 'belief_tracker'):
                    dropped_indices = [ALL_CARDS.index(c) for c in self.talon]
                    self.belief_tracker.drop_talon(self.current_player, dropped_indices)
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        elif self.phase == "bidding":
            if action == 45:
                from engine.trick import ALL_CARDS
                if hasattr(self, 'belief_tracker'):
                    robbed_indices = [ALL_CARDS.index(c) for c in self.talon]
                    self.belief_tracker.rob_talon(self.current_player, robbed_indices)
                    
                for c in self.talon:
                    idx = ALL_CARDS.index(c)
                    self.hands[self.current_player][idx] = 1
                self.talon = []
                self.phase = "drop_talon"
                self.cards_to_drop = 2
                self.must_bid_higher = True
                self.has_robbed[self.current_player] = True
                return self._get_obs(), reward, terminated, truncated, self._get_info()

            bidding_mask = self.auction.get_action_mask()
            # We don't check bidding_mask[action] here because our info mask already verified it
                
            self.auction.step(action)
            self.must_bid_higher = False # Reset it
            
            if self.auction.is_over:
                if self.auction.highest_bid is None:
                    self.auction.highest_bidder = self.auction.starting_player
                    self.auction.highest_bid = ALL_BIDS[0] # Fallback to Passz game
                
                bid = self.auction.highest_bid
                if bid.is_piros:
                    self.trump_suit = Suit.HEARTS
                    self._setup_playing_phase()
                elif bid.is_betli or bid.is_durchmars:
                    self.trump_suit = None
                    self._setup_playing_phase()
                else:
                    self.phase = "pick_trump"
                    self.current_player = self.auction.highest_bidder
            else:
                self.current_player = self.auction.active_player
                
        elif self.phase == "pick_trump":
            if action == 40:
                self.trump_suit = Suit.ACORNS
            elif action == 41:
                self.trump_suit = Suit.LEAVES
            elif action == 42:
                self.trump_suit = Suit.BELLS
            else:
                raise ValueError(f"Invalid trump pick {action}")
            self._setup_playing_phase()

        elif self.phase == "playing":
            if action >= 32:
                raise ValueError(f"Invalid play action {action}")
                
            mask = get_action_mask(self.hands[self.current_player], self.trick)
            if not mask[action]:
                raise ValueError(f"Invalid play action {action}")
                
            card = ALL_CARDS[action]
            self.hands[self.current_player][action] = 0
            
            # Neuro-symbolic belief tracking BEFORE trick state updates
            if hasattr(self, 'belief_tracker'):
                self.belief_tracker.play_card(self.current_player, action)
                if self.trick.lead_suit:
                    if card.suit != self.trick.lead_suit:
                        self.belief_tracker.mark_void(self.current_player, self.trick.lead_suit)
                        # Furthermore, if they didn't play trump, they might be void in trump too!
                        if card.suit != self.trump_suit and self.trump_suit is not None:
                            self.belief_tracker.mark_void(self.current_player, self.trump_suit)
                    else:
                        # They followed suit. Did they overtrick?
                        highest_lead = max([c for c in self.trick.cards_played if c.suit == self.trick.lead_suit], key=lambda x: self.trick.rank_power[x.rank], default=None)
                        if highest_lead and self.trick.rank_power[card.rank] < self.trick.rank_power[highest_lead.rank]:
                            self.belief_tracker.mark_cannot_overtrick(self.current_player, self.trick.lead_suit, highest_lead.rank.value)
            
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
                            
                    if bid.id == 0:
                        reward = 0.5 if declarer_won else -1.0
                    elif bid.id == 1: # Piros passz
                        reward = float(bid.points) if declarer_won else float(-bid.points)
                    else:
                        reward = float(bid.points * 2) if declarer_won else float(-bid.points)
                
                self.current_player = winner
                if not terminated:
                    self.trick = Trick(trump_suit=self.trump_suit, is_betli_or_durchmars=bid.is_betli or bid.is_durchmars)
            else:
                self.current_player = (self.current_player + 1) % 3

        return self._get_obs(), reward, terminated, truncated, self._get_info()
