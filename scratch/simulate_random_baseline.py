import random
import numpy as np
from engine.environments.ulti import UltiEnv
from engine.bidding import BIDS_BY_ID

def run_random_baseline(num_episodes=10000):
    env = UltiEnv()
    
    # Tracking
    mode_eps = {"normal": 0, "betli": 0, "durchmars": 0, "ulti": 0}
    mode_wins = {"normal": 0, "betli": 0, "durchmars": 0, "ulti": 0}
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # Get valid actions
            mask = info["action_mask"]
            valid_actions = np.where(mask)[0]
            
            if len(valid_actions) == 0:
                print("Error: No valid actions!")
                break
                
            # Take a completely random valid action
            action = random.choice(valid_actions)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
        # Game over, record results
        bid = env.auction.highest_bid
        mode = "normal"
        if bid is not None and bid.id != 0:
            if bid.is_durchmars:
                mode = "durchmars"
            elif bid.is_betli:
                mode = "betli"
            elif bid.has_ulti:
                mode = "ulti"
                
        mode_eps[mode] += 1
        
        # In our environment:
        # Passz (+0.5 for win, -0.5 for loss)
        # Bids (reward > 0 means win)
        if reward > 0:
            mode_wins[mode] += 1
            
    print(f"--- RANDOM BASELINE SIMULATION ({num_episodes:,} Games) ---")
    for m in ["normal", "betli", "durchmars", "ulti"]:
        if mode_eps[m] > 0:
            win_rate = (mode_wins[m] / mode_eps[m]) * 100
            dist = (mode_eps[m] / num_episodes) * 100
            print(f"{m.capitalize()}:")
            print(f"  Distribution: {dist:.2f}%")
            print(f"  Win Rate:     {win_rate:.2f}%")
            print(f"  Total Played: {mode_eps[m]:,}")
        else:
            print(f"{m.capitalize()}: 0 played")
            
if __name__ == "__main__":
    run_random_baseline()
