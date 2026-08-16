import os
import json
import time
import copy
import random
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from engine.environments.ulti import UltiEnv
from agent.ppo import PPOMultiHeadAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_hyperparams(path: str = "config/hyperparams.json") -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load hyperparams: {e}")
        return None

def write_telemetry(path: str, data: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")

class ReplayBuffer:
    def __init__(self) -> None:
        self.reset()
        
    def reset(self) -> None:
        self.obs: List[Dict[str, np.ndarray]] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.is_declarer: List[float] = []
        self.action_masks: List[np.ndarray] = []
        self.modes: List[str] = []

    def store(self, obs: Dict[str, np.ndarray], action: int, logprob: float, reward: float, value: float, done: bool, is_dec: float, mask: np.ndarray, mode: str) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.is_declarer.append(is_dec)
        self.action_masks.append(mask)
        self.modes.append(mode)

def compute_advantages(rewards: List[float], values: List[float], dones: List[bool], gamma: float, gae_lambda: float) -> List[float]:
    advantages: List[float] = []
    gae = 0
    values = values + [0.0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages

def stack_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    stacked: Dict[str, torch.Tensor] = {}
    for k in obs_list[0].keys():
        stacked[k] = torch.tensor(np.stack([o[k] for o in obs_list]), device=device)
    return stacked

def update_agent(agent: PPOMultiHeadAgent, optimizer: optim.Optimizer, buffer: ReplayBuffer, params: Dict[str, Any]) -> Tuple[float, float, float]:
    obs_batch = stack_obs(buffer.obs)
    actions = torch.tensor(buffer.actions, device=device)
    old_logprobs = torch.tensor(buffer.logprobs, device=device)
    returns = torch.tensor(compute_advantages(buffer.rewards, buffer.values, buffer.dones, params["gamma"], params["gae_lambda"]), device=device) + torch.tensor(buffer.values, device=device)
    advantages = returns - torch.tensor(buffer.values, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    is_declarers = torch.tensor(buffer.is_declarer, dtype=torch.float32, device=device).unsqueeze(1)
    masks = torch.tensor(np.array(buffer.action_masks), dtype=torch.bool, device=device)
    
    dataset_size = len(buffer.obs)
    batch_size = params["batch_size"]
    
    for _ in range(params["ppo_epochs"]):
        indices = np.random.permutation(dataset_size)
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            idx = indices[start_idx:end_idx]
            
            # Since mode is categorical (string), we iterate inside or group by mode.
            # For simplicity in this demo loop, we can evaluate one by one or mask them.
            # To vectorize across modes, we can compute all logits and select.
            batch_obs = {k: v[idx] for k, v in obs_batch.items()}
            batch_is_dec = is_declarers[idx]
            batch_masks = masks[idx]
            batch_actions = actions[idx]
            batch_old_logprobs = old_logprobs[idx]
            batch_adv = advantages[idx].float()
            batch_returns = returns[idx].float()
            
            modes_batch = [buffer.modes[i] for i in idx]
            
            # Calculate new logprobs, entropy, values
            new_logprobs = []
            entropies = []
            values_list = []
            
            # We process individually since mode might differ, or group by mode
            # Optimization: could group by mode, but for robust loop we'll do simple inference
            for i in range(len(idx)):
                obs_i = {k: v[i] for k, v in batch_obs.items()}
                mode_i = modes_batch[i]
                mask_i = batch_masks[i]
                act_i = batch_actions[i]
                dec_i = batch_is_dec[i]
                
                _, logprob, entropy, val = agent.get_action_and_value(
                    obs_i, dec_i, mode=mode_i, action_mask=mask_i, action=act_i
                )
                new_logprobs.append(logprob)
                entropies.append(entropy)
                values_list.append(val)
                
            new_logprobs = torch.stack(new_logprobs).squeeze(-1)
            entropies = torch.stack(entropies).squeeze(-1)
            values = torch.stack(values_list).squeeze(-1)
            
            ratio = torch.exp(new_logprobs - batch_old_logprobs)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1.0 - params["clip_ratio"], 1.0 + params["clip_ratio"]) * batch_adv
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, batch_returns)
            entropy_loss = entropies.mean()
            
            loss = actor_loss + 0.5 * critic_loss - params["entropy_coef"] * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
    return actor_loss.item(), critic_loss.item(), entropy_loss.item()

def train() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    writer = SummaryWriter(log_dir=f"logs/tb/run_{int(time.time())}")
    
    hyperparams_file = "config/hyperparams.json"
    params = load_hyperparams(hyperparams_file)
    last_reload = time.time()
    
    agent = PPOMultiHeadAgent().to(device)
    if os.path.exists("models/agent_checkpoint.pth"):
        agent.load_state_dict(torch.load("models/agent_checkpoint.pth", weights_only=True, map_location=device))
        print("Loaded existing checkpoint!")
    optimizer = optim.Adam(agent.parameters(), lr=params["learning_rate"])
    
    historical_weights = deque(maxlen=params["fictitious_play_history_size"])
    
    # Defenders might use older weights for Fictitious Play
    defender_agents = [PPOMultiHeadAgent().to(device), PPOMultiHeadAgent().to(device)]
    for da in defender_agents:
        da.load_state_dict(agent.state_dict())
        da.eval()
    
    env = UltiEnv()
    buffer = ReplayBuffer()
    
    global_step = 0
    episodes = 0
    wins = 0
    total_reward = 0.0
    from collections import defaultdict
    bid_counts = defaultdict(int)
    bid_wins = defaultdict(int)
    player_trajectories = {0: [], 1: [], 2: []}
    
    obs, info = env.reset()
    
    while True:
        # Hot-reload hyperparams every 10 seconds
        if time.time() - last_reload > 10:
            new_params = load_hyperparams(hyperparams_file)
            if new_params:
                params = new_params
                # Update learning rate if changed
                for param_group in optimizer.param_groups:
                    param_group['lr'] = params["learning_rate"]
            last_reload = time.time()
            
        current_player = env.current_player
        is_declarer = 1.0 if env.auction.highest_bidder == current_player else 0.0
        
        # Decide mode based on highest bid
        mode = "normal"
        if env.auction.highest_bid is not None:
            bid_val = env.auction.highest_bid
            if bid_val.is_durchmars:
                mode = "durchmars"
            elif bid_val.is_betli:
                mode = "betli"
                
        action_mask = torch.tensor(info["action_mask"], dtype=torch.bool, device=device)
        
        # Fictitious Play for defenders
        use_agent = agent
        if not is_declarer and random.random() < params["fictitious_play_prob"] and len(historical_weights) > 0:
            use_agent = random.choice(defender_agents)
            
        with torch.no_grad():
            action, logprob, entropy, value = use_agent.get_action_and_value(
                obs, is_declarer, mode=mode, action_mask=action_mask
            )
            
        current_player = env.current_player
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        
        if use_agent == agent:
            # Store temporarily for this player
            player_trajectories[current_player].append((obs, action.item(), logprob.item(), value.item(), is_declarer, action_mask.cpu().numpy(), mode))
            
        obs = next_obs
        global_step += 1
        
        if terminated or truncated:
            # Flush trajectories to buffer with precise perspective rewards
            for p_id, traj in player_trajectories.items():
                for i, transition in enumerate(traj):
                    obs_t, action_t, logprob_t, value_t, is_declarer_t, mask_t, mode_t = transition
                    
                    if i == len(traj) - 1:
                        # Only give the terminal reward to the final step
                        step_reward = reward if is_declarer_t == 1.0 else -reward
                        step_done = True
                    else:
                        step_reward = 0.0
                        step_done = False
                        
                    buffer.store(obs_t, action_t, logprob_t, step_reward, value_t, step_done, is_declarer_t, mask_t, mode_t)
            
            player_trajectories = {0: [], 1: [], 2: []}
            
            episodes += 1
            if env.auction.highest_bid is None:
                bid_name = "pass"
            else:
                bid_name = env.auction.highest_bid.name
            
            bid_counts[bid_name] += 1
            if reward > 0:
                wins += 1
                bid_wins[bid_name] += 1
                
            obs, info = env.reset()
            
        if len(buffer.obs) >= params["update_frequency"]:
            a_loss, c_loss, e_loss = update_agent(agent, optimizer, buffer, params)
            buffer.reset()
            
            win_rate = wins / max(1, episodes)
            
            # Telemetry
            telemetry_data = {
                "step": global_step,
                "actor_loss": a_loss,
                "critic_loss": c_loss,
                "entropy": e_loss,
                "win_rate": win_rate
            }
            write_telemetry("logs/telemetry.jsonl", telemetry_data)
            
            # TensorBoard
            writer.add_scalar("Loss/Actor", a_loss, global_step)
            writer.add_scalar("Loss/Critic", c_loss, global_step)
            writer.add_scalar("Loss/Entropy", e_loss, global_step)
            writer.add_scalar("Metrics/WinRate", win_rate, global_step)
            writer.add_scalar("Metrics/AverageReward", total_reward / max(1, episodes), global_step)
            if bid_counts:
                labels = list(bid_counts.keys())
                sizes = list(bid_counts.values())
                fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax_pie.axis('equal')
                writer.add_figure("Charts/Bid_Distribution", fig_pie, global_step)
                
                rates = [bid_wins[k] / max(1, bid_counts[k]) for k in labels]
                fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                ax_bar.bar(labels, rates)
                ax_bar.set_ylim(0, 1.0)
                ax_bar.set_ylabel('Win Rate')
                ax_bar.set_title('Win Rate per Bid Type')
                plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                fig_bar.tight_layout()
                writer.add_figure("Charts/Bid_WinRates", fig_bar, global_step)
            
            print(f"Step: {global_step} | Win Rate: {win_rate:.2f} | Avg Reward: {(total_reward / max(1, episodes)):.2f} | A_Loss: {a_loss:.4f} | C_Loss: {c_loss:.4f}")
            torch.save(agent.state_dict(), 'models/agent_checkpoint.pth')
            
            # Store checkpoint for Fictitious Play
            historical_weights.append(copy.deepcopy(agent.state_dict()))
            if len(historical_weights) > 0:
                # Update defender agents with historical weights
                for da in defender_agents:
                    hist_idx = random.randint(0, len(historical_weights) - 1)
                    da.load_state_dict(historical_weights[hist_idx])
                    
            # Reset counters
            episodes = 0
            wins = 0
            total_reward = 0.0
            bid_counts.clear()
            bid_wins.clear()

if __name__ == "__main__":
    train()
