import sys
import numpy as np
import random
from engine.environments.ulti import UltiEnv
from engine.bidding import ALL_BIDS, BIDS_BY_ID
from engine.trick import ALL_CARDS

def print_separator():
    print("=" * 60)

def main():
    env = UltiEnv()
    obs, info = env.reset()
    
    print("Welcome to the Rablóulti Human Test Interface!")
    print("You are Player 0. P1 and P2 are random bots.")
    
    terminated = False
    
    while not terminated:
        print_separator()
        phase = env.phase
        current_player = env.current_player
        
        # Current state info
        print(f"Phase: {phase.upper()} | Current Player: P{current_player}")
        if phase == "playing":
            print(f"Trump Suit: {env.trump_suit.name if env.trump_suit else 'None'}")
            if env.auction.highest_bid:
                print(f"Highest Bid: {env.auction.highest_bid.name} (by P{env.auction.highest_bidder})")
            print(f"Trick #{env.tricks_played + 1} | Declarer Points: {env.declarer_points} | Defenders Points: {env.defenders_points}")
            
            trick_cards = env.trick.cards_played
            if trick_cards:
                print("Current Trick:")
                for p_id, card in zip(env.trick.players, trick_cards):
                    print(f"  P{p_id} played: {card}")
            else:
                print("Current Trick: Empty")
                
        # Player 0's perspective
        if current_player == 0:
            print("\nYour Hand:")
            hand_ids = np.where(obs["hand"])[0]
            for idx in hand_ids:
                print(f"  [{idx}] {ALL_CARDS[idx]}")
                
            mask = info["action_mask"]
            legal_actions = np.where(mask)[0]
            
            print("\nLegal Actions:")
            for act in legal_actions:
                if phase == "bidding":
                    bid = BIDS_BY_ID[act]
                    print(f"  [{act}] {bid.name} ({bid.points} pts)")
                else:
                    print(f"  [{act}] Play {ALL_CARDS[act]}")
                    
            while True:
                try:
                    choice = int(input(f"\nEnter your choice for P0: "))
                    if choice in legal_actions:
                        break
                    print("Invalid choice. Try again.")
                except ValueError:
                    print("Please enter a valid integer.")
                    
            action = choice
        else:
            # Bot makes a random move
            mask = info["action_mask"]
            legal_actions = np.where(mask)[0]
            action = random.choice(legal_actions)
            
            if phase == "bidding":
                bid = BIDS_BY_ID[action]
                print(f"P{current_player} bids: {bid.name}")
            else:
                card = ALL_CARDS[action]
                print(f"P{current_player} plays: {card}")
                
        obs, reward, terminated, truncated, info = env.step(action)
        
    print_separator()
    print("GAME OVER")
    print(f"Reward: {reward}")
    print(f"Final Declarer Points: {env.declarer_points}")
    print(f"Final Defenders Points: {env.defenders_points}")
    print(f"Declarer Tricks Won: {env.declarer_tricks_won}")
    print(f"Defenders Tricks Won: {env.defenders_tricks_won}")
    if env.auction.highest_bid:
        print(f"Contract was: {env.auction.highest_bid.name} by P{env.auction.highest_bidder}")

if __name__ == "__main__":
    main()
