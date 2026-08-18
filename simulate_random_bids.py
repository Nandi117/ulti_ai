import sys
import numpy as np
from engine.environments.ulti import UltiEnv
from engine.bidding import ALL_BIDS
from collections import defaultdict

def main():
    print("Initializing environment...")
    # Initialize environment with curriculum_mode=False to allow all 10 bids
    env = UltiEnv(curriculum_mode=False)
    
    num_episodes = 50000
    
    mode_counts = defaultdict(int)
    
    print(f"Running {num_episodes} purely random simulations to test Bidding distribution...")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        
        while True:
            mask = info["action_mask"]
            legal_actions = [i for i, m in enumerate(mask) if m]
            action = np.random.choice(legal_actions)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                # Get the final bid
                if env.auction.highest_bid is not None and env.auction.highest_bid.id != 0:
                    mode = env.auction.highest_bid.name
                else:
                    mode = "Passz"
                mode_counts[mode] += 1
                break
                
        if (ep + 1) % 5000 == 0:
            print(f"Progress: {ep + 1}/{num_episodes}")

    print("\nFINAL RANDOM BIDDING DISTRIBUTION:")
    for b in ALL_BIDS:
        name = b.name.capitalize()
        if name == "Passz":
            name = "Passz"
        count = mode_counts.get(name, 0)
        print(f"{name:15} : {count}")
        
    print(f"\nTotal Episodes: {sum(mode_counts.values())}")

if __name__ == "__main__":
    main()
