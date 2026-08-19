import os
import torch
import random
import numpy as np
from collections import defaultdict
from engine.environments.ulti import UltiEnv
from agent.ppo import PPOMultiHeadAgent
from agent.baselines.heuristic import HeuristicAgent

def load_agents(device):
    declarer_agent = PPOMultiHeadAgent().to(device)
    defender_agent = PPOMultiHeadAgent().to(device)
    
    checkpoint_path = r'C:\ulti_ai\models\best_model_exceptional_hand.pth'
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        declarer_agent.load_state_dict(state_dict['declarer'])
        defender_agent.load_state_dict(state_dict['defender'])
        print(f"Model loaded successfully from {checkpoint_path}")
    else:
        print("Model checkpoint not found!")
    declarer_agent.eval()
    defender_agent.eval()
    return declarer_agent, defender_agent

def play_games(declarer_agent, defender_agent, env, device, num_games=1000, opponents="agent"):
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
                    is_declarer = 1.0 if (env.auction and env.auction.highest_bidder == current_player) else 0.0
                    if env.auction and env.auction.highest_bidder is None:
                        is_declarer = 0.0
                        
                    if env.phase == "talon":
                        mode = "talon"
                        active_agent = declarer_agent
                    elif env.phase == "decision_to_rob" or env.phase == "bidding":
                        mode = "decision_to_rob" if env.phase == "decision_to_rob" else "normal"
                        active_agent = declarer_agent
                    else:
                        bid = env.auction.highest_bid
                        if bid is not None and bid.id != 0:
                            if bid.is_betli:
                                mode = "betli"
                            elif bid.is_durchmars:
                                mode = "durchmars"
                            elif bid.has_ulti:
                                mode = "ulti"
                            else:
                                mode = "normal"
                        else:
                            mode = "normal"
                            
                        active_agent = declarer_agent if is_declarer == 1.0 else defender_agent
                            
                    action, _, _, _ = active_agent.get_action_and_value(
                        obs, mode=mode, action_mask=torch.tensor(mask, device=device)
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
