import os
import torch
import random
import numpy as np
from collections import defaultdict
from engine.environments.ulti import UltiEnv
from agent.ppo import PPOMultiHeadAgent
from agent.baselines.heuristic import HeuristicAgent

def load_agent(device):
    agent = PPOMultiHeadAgent().to(device)
    checkpoint_path = r'C:\ulti_ai\models\agent_checkpoint_belief_tracker_final.pth'
    if os.path.exists(checkpoint_path):
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print("Model checkpoint not found!")
    agent.eval()
    return agent

def play_games(agent, env, device, num_games=1000, opponents="agent"):
    # opponents can be "agent", "random", or "heuristic"
    heuristic_bot = HeuristicAgent()
    
    wins = 0
    total_played = 0
    bids = defaultdict(int)
    
    for game_idx in range(num_games):
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            current_player = env.current_player
            mask = info["action_mask"]
            legal_actions = np.where(mask)[0]
            
            if current_player == 0 or opponents == "agent":
                # Neural Network Decision
                with torch.no_grad():
                    is_declarer = 1.0 if env.auction.highest_bidder == current_player else 0.0
                    
                    if env.phase == "drop_talon":
                        mode = "Talon"
                    elif env.phase == "decision_to_rob" or env.phase == "bidding":
                        mode = "decision_to_rob" if env.phase == "decision_to_rob" else "normal"
                    else:
                        contract = env.auction.highest_bid.name.lower() if env.auction.highest_bid else "passz"
                        if "betli" in contract:
                            mode = "betli"
                        elif "durchmars" in contract:
                            mode = "durchmars"
                        else:
                            mode = "normal"
                            
                    action, _, _, _ = agent.get_action_and_value(
                        obs, is_declarer, mode=mode, action_mask=torch.tensor(mask, device=device)
                    )
                    action = action.item()
            elif opponents == "random":
                action = random.choice(legal_actions)
            elif opponents == "heuristic":
                action = heuristic_bot.act(obs, mask)
                    
            obs, reward, terminated, truncated, info = env.step(action)
            
        if env.auction.highest_bid:
            bids[env.auction.highest_bid.name] += 1
            if env.auction.highest_bidder == 0:
                total_played += 1
                if reward > 0:
                    wins += 1
                    
    win_rate = (wins / total_played * 100) if total_played > 0 else 0
    
    result_data = {
        "opponents": opponents,
        "total_games": num_games,
        "games_as_declarer": total_played,
        "wins": wins,
        "win_rate": win_rate,
        "bidding_distribution": dict(bids)
    }
    
    print(f"\nOpponents: {opponents.upper()}")
    print(f"Games as Declarer (Player 0): {total_played}/{num_games}")
    print(f"Wins: {wins}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Bidding Distribution: {dict(bids)}")
    return result_data

if __name__ == "__main__":
    import json
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = load_agent(device)
    
    results = []
    
    # 1. Agent vs Agent (Self-play)
    print("Testing Agent vs Agent...")
    env_self = UltiEnv(curriculum_mode=False, training_filter_mode=True)
    results.append(play_games(agent, env_self, device, num_games=1000, opponents="agent"))
    
    # 2. Agent vs Random
    print("Testing Agent vs Random...")
    env_random = UltiEnv(curriculum_mode=False, training_filter_mode=True)
    results.append(play_games(agent, env_random, device, num_games=1000, opponents="random"))
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
