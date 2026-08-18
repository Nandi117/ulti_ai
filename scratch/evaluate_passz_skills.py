import torch
import numpy as np
from engine.environments.ulti import UltiEnv
from engine.core import Deck
from agent.ppo import PPOMultiHeadAgent

def evaluate_passz_skills():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOMultiHeadAgent().to(device)
    try:
        agent.load_state_dict(torch.load("models/agent_checkpoint.pth", map_location=device))
        print("Loaded trained agent.")
    except Exception as e:
        print(f"Error loading agent: {e}")
        return
        
    agent.eval()
    env = UltiEnv()
    
    num_episodes = 1000
    wins = 0
    
    print(f"Forcing the trained agent to play {num_episodes} games of Passz on completely random hands...")
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            mask = info["action_mask"]
            is_declarer = 1.0 if (env.auction and env.current_player == env.auction.highest_bidder) else 0.0
            phase = 3 if (env.auction and env.auction.is_over) else 1
            
            if phase == 1:
                # Force everyone to Passz!
                action_item = 0 
            else:
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(
                        obs, is_declarer, mode="normal", action_mask=torch.tensor(mask, device=device)
                    )
                    action_item = action.item()
                    
            obs, reward, terminated, truncated, info = env.step(action_item)
            
        if reward > 0:
            wins += 1
            
    win_rate = (wins / num_episodes) * 100
    print(f"--- FORCED PASSZ RESULTS ---")
    print(f"Games Played: {num_episodes}")
    print(f"Agent's True Passz Win Rate (on average hands): {win_rate:.2f}%")
    print(f"Random Baseline Win Rate (from earlier): ~40.14%")

if __name__ == "__main__":
    evaluate_passz_skills()
