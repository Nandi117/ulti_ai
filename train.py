import os
import time
import json
import torch
import random
import numpy as np
import copy
from collections import defaultdict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import concurrent.futures

from engine.environments.ulti import UltiEnv
from agent.ppo import PPOMultiHeadAgent
from agent.baselines.heuristic import HeuristicAgent

def load_hyperparams(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load hyperparams: {e}")
        return None

def write_telemetry(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(json.dumps(data) + '\n')

class ParallelReplayBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.is_declarer = []
        self.action_masks = []
        self.modes = []

    def extend(self, trajectories):
        for t in trajectories:
            self.obs.append(t[0])
            self.actions.append(t[1])
            self.logprobs.append(t[2])
            self.rewards.append(t[3])
            self.values.append(t[4])
            self.dones.append(t[5])
            self.is_declarer.append(t[6])
            self.action_masks.append(t[7])
            self.modes.append(t[8])

    def reset(self):
        self.__init__()

def update_agent(agent, optimizer, buffer, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    b_obs = {
        "hand": torch.tensor(np.array([o["hand"] for o in buffer.obs]), dtype=torch.float32, device=device),
        "trick_history": torch.tensor(np.array([o["trick_history"] for o in buffer.obs]), dtype=torch.float32, device=device),
        "deduction_flags": torch.tensor(np.array([o["deduction_flags"] for o in buffer.obs]), dtype=torch.float32, device=device),
        "trump_suit": torch.tensor(np.array([o["trump_suit"] for o in buffer.obs]), dtype=torch.float32, device=device),
        "lead_suit": torch.tensor(np.array([o["lead_suit"] for o in buffer.obs]), dtype=torch.float32, device=device),
        "scores": torch.tensor(np.array([o["scores"] for o in buffer.obs]), dtype=torch.float32, device=device),
        "belief_state": torch.tensor(np.array([o["belief_state"] for o in buffer.obs]), dtype=torch.float32, device=device)
    }
    
    b_actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    b_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32, device=device)
    b_rewards = torch.tensor(buffer.rewards, dtype=torch.float32, device=device)
    b_values = torch.tensor(buffer.values, dtype=torch.float32, device=device)
    b_is_declarer = torch.tensor(buffer.is_declarer, dtype=torch.float32, device=device)
    b_masks = torch.tensor(np.array(buffer.action_masks), dtype=torch.bool, device=device)
    
    returns = []
    discounted_reward = 0
    for reward, is_done in zip(reversed(b_rewards.cpu().numpy()), reversed(buffer.dones)):
        if is_done:
            discounted_reward = 0
        discounted_reward = reward + params["gamma"] * discounted_reward
        returns.insert(0, discounted_reward)
        
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = returns - b_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_a_loss, total_c_loss, total_e_loss = 0, 0, 0
    
    for epoch in range(params.get("ppo_epochs", 4)):
        for mode in ["normal", "betli", "durchmars", "ulti"]:
            indices = [i for i, m in enumerate(buffer.modes) if m == mode]
            if not indices: continue
            
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            
            m_obs = {
                "hand": b_obs["hand"][idx_tensor],
                "trick_history": b_obs["trick_history"][idx_tensor],
                "deduction_flags": b_obs["deduction_flags"][idx_tensor],
                "trump_suit": b_obs["trump_suit"][idx_tensor],
                "lead_suit": b_obs["lead_suit"][idx_tensor],
                "scores": b_obs["scores"][idx_tensor],
                "belief_state": b_obs["belief_state"][idx_tensor]
            }
            m_declarer = b_is_declarer[idx_tensor]
            m_actions = b_actions[idx_tensor]
            m_logprobs = b_logprobs[idx_tensor]
            m_advantages = advantages[idx_tensor]
            m_returns = returns[idx_tensor]
            m_masks = b_masks[idx_tensor]
            
            probs, values = agent(m_obs, m_declarer, mode=mode, action_mask=m_masks)
            
            # Re-apply mask to ensure invalid actions stay exactly 0
            probs = probs * m_masks.bool()
            probs = probs + 1e-10
            probs = probs * m_masks.bool()
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            from torch.distributions.categorical import Categorical
            dist = Categorical(probs=probs)
            
            new_logprobs = dist.log_prob(m_actions)
            entropy = dist.entropy().mean()
            
            logratio = new_logprobs - m_logprobs
            ratio = logratio.exp()
            
            pg_loss1 = -m_advantages * ratio
            clip_ratio = params.get("clip_ratio", 0.2)
            pg_loss2 = -m_advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            actor_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            critic_loss = 0.5 * ((m_returns - values.squeeze(-1)) ** 2).mean()
            
            loss = actor_loss - params.get("entropy_coef", 0.01) * entropy + critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
            total_a_loss += actor_loss.item()
            total_c_loss += critic_loss.item()
            total_e_loss += entropy.item()
            
    return total_a_loss, total_c_loss, total_e_loss


def worker_episodes(state_dict, params, phase, num_episodes=5):
    agent = PPOMultiHeadAgent()
    agent.load_state_dict(state_dict)
    agent.eval()
    
    heuristic = HeuristicAgent()
    env = UltiEnv(curriculum_mode=True)
    
    collected_trajectories = []
    
    stats = {
        "episodes": 0,
        "wins": 0,
        "mode_episodes": defaultdict(int),
        "mode_wins": defaultdict(int)
    }
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_traj = {0: [], 1: [], 2: []}
        
        while True:
            current_player = env.current_player
            is_declarer = 1.0 if env.auction.highest_bidder == current_player else 0.0
            
            if env.phase == "drop_talon":
                mode = "talon"
            elif env.phase == "decision_to_rob" or env.phase == "bidding":
                mode = "decision_to_rob" if env.phase == "decision_to_rob" else "normal" # Default bidding to normal head for now, wait we could use normal
            else:
                mode = "Passz"
                bid = env.auction.highest_bid
                if bid is not None and bid.id != 0:
                    mode = bid.name
            
            mask = info["action_mask"]
            legal_actions = [i for i, m in enumerate(mask) if m]
            
            action_item = None
            logprob_item = 0.0
            value_item = 0.0
            
            if is_declarer == 1.0:
                with torch.no_grad():
                    action, logprob, entropy, value = agent.get_action_and_value(
                        obs, is_declarer, mode=mode if mode != "ulti" else "normal", action_mask=torch.tensor(mask)
                    )
                    action_item = action.item()
                    logprob_item = logprob.item()
                    value_item = value.item()
            else:
                if phase == 1:
                    action_item = random.choice(legal_actions)
                elif phase == 2:
                    action_item = heuristic.act(obs, mask)
                else: 
                    with torch.no_grad():
                        action, logprob, entropy, value = agent.get_action_and_value(
                            obs, is_declarer, mode=mode if mode != "ulti" else "normal", action_mask=torch.tensor(mask)
                        )
                        action_item = action.item()
                        logprob_item = logprob.item()
                        value_item = value.item()
            
            next_obs, reward, terminated, truncated, info = env.step(action_item)
            
            if is_declarer == 1.0 or phase == 3:
                episode_traj[current_player].append((obs, action_item, logprob_item, value_item, is_declarer, mask, mode))
                
            obs = next_obs
            
            if terminated or truncated:
                stats["episodes"] += 1
                stats["mode_episodes"][mode] += 1
                if reward > 0:
                    stats["wins"] += 1
                    stats["mode_wins"][mode] += 1
                    
                for p_id, traj in episode_traj.items():
                    for i, transition in enumerate(traj):
                        obs_t, action_t, logprob_t, value_t, is_declarer_t, mask_t, mode_t = transition
                        if i == len(traj) - 1:
                            step_reward = reward if is_declarer_t == 1.0 else -reward
                            step_done = True
                        else:
                            step_reward = 0.0
                            step_done = False
                        
                        collected_trajectories.append((obs_t, action_t, logprob_t, step_reward, value_t, step_done, is_declarer_t, mask_t, mode_t))
                break
                
    return collected_trajectories, dict(stats)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Multi-Phase Curriculum Training on {device} (Highly Optimized Single-Threaded)...")
    
    hyperparams_file = "config/hyperparams.json"
    params = load_hyperparams(hyperparams_file)
    
    agent = PPOMultiHeadAgent().to(device)
    
    # Load Phase 1 Checkpoint
    checkpoint_path = r'C:\ulti_ai\models\agent_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print("Successfully loaded Phase 1 checkpoint! Starting Phase 2.")
    else:
        print("No checkpoint found. Starting from scratch.")
        
    optimizer = optim.Adam(agent.parameters(), lr=params["learning_rate"], eps=1e-5)
    
    import time
    run_name = f"bidding_phase_{int(time.time())}"
    writer = SummaryWriter(f"logs/tb/{run_name}")
    print(f"Tensorboard logs saving to logs/tb/{run_name}")
    
    buffer = ParallelReplayBuffer()
    heuristic = HeuristicAgent()
    
    # Dynamic Action Masking: No curriculum (no rigged hands), but logically impossible/terrible bids are masked out
    env = UltiEnv(curriculum_mode=False, training_filter_mode=True)
    
    global_step = 0
    phase = 3
    recent_win_rates = []
    
    batch_episodes = 0
    batch_wins = 0
    batch_rewards = 0.0
    mode_eps = defaultdict(int)
    mode_wins = defaultdict(int)
    
    obs, info = env.reset()
    episode_traj = {0: [], 1: [], 2: []}
    last_print = time.time()
    
    while True:
        current_player = env.current_player
        is_declarer = 1.0 if env.auction.highest_bidder == current_player else 0.0
        
        if env.phase == "drop_talon":
            dash_mode = "Talon"
            nn_mode = "talon"
        elif env.phase == "decision_to_rob" or env.phase == "bidding":
            dash_mode = "Robbing" if env.phase == "decision_to_rob" else "Bidding"
            nn_mode = "decision_to_rob" if env.phase == "decision_to_rob" else "normal"
        else:
            dash_mode = "Passz"
            nn_mode = "normal"
            bid = env.auction.highest_bid
            if bid is not None and bid.id != 0:
                dash_mode = bid.name.split(" (")[0]
                if bid.is_betli:
                    nn_mode = "betli"
                elif bid.is_durchmars:
                    nn_mode = "durchmars"
                elif bid.has_ulti:
                    nn_mode = "ulti"
                else:
                    nn_mode = "normal"
            
        if dash_mode not in mode_eps:
            mode_eps[dash_mode] = 0
        if dash_mode not in mode_wins:
            mode_wins[dash_mode] = 0
            
        mask = info["action_mask"]
        legal_actions = [i for i, m in enumerate(mask) if m]
        
        action_item = None
        logprob_item = 0.0
        value_item = 0.0
        
        if is_declarer == 1.0:
            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(
                    obs, is_declarer, mode=nn_mode, action_mask=torch.tensor(mask, device=device)
                )
                if env.phase == "bidding":
                    from engine.bidding import ALL_BIDS
                    bid_made = ALL_BIDS[action.item()]
                    print(f"AGENT BID: {bid_made.name} (ID: {bid_made.id})")
                action_item = action.item()
                logprob_item = logprob.item()
                value_item = value.item()
        else:
            if phase == 1:
                action_item = random.choice(legal_actions)
            elif phase == 2:
                action_item = heuristic.act(obs, mask)
            else:
                with torch.no_grad():
                    action, logprob, entropy, value = agent.get_action_and_value(
                        obs, is_declarer, mode=nn_mode, action_mask=torch.tensor(mask, device=device)
                    )
                    action_item = action.item()
                    logprob_item = logprob.item()
                    value_item = value.item()
        
        next_obs, reward, terminated, truncated, info = env.step(action_item)
        
        if is_declarer == 1.0 or phase == 3:
            episode_traj[current_player].append((obs, action_item, logprob_item, value_item, is_declarer, mask, nn_mode))
            
        obs = next_obs
        global_step += 1
        
        if terminated or truncated:
            if env.auction.highest_bid is None:
                dash_mode = "Passz"
                
            batch_episodes += 1
            batch_rewards += reward
            mode_eps[dash_mode] += 1
            if reward > 0:
                batch_wins += 1
                mode_wins[dash_mode] += 1
                
            collected_trajectories = []
            for p_id, traj in episode_traj.items():
                for i, transition in enumerate(traj):
                    obs_t, action_t, logprob_t, value_t, is_declarer_t, mask_t, mode_t = transition
                    if i == len(traj) - 1:
                        step_reward = reward if is_declarer_t == 1.0 else -reward
                        step_done = True
                    else:
                        step_reward = 0.0
                        step_done = False
                    
                    collected_trajectories.append((obs_t, action_t, logprob_t, step_reward, value_t, step_done, is_declarer_t, mask_t, mode_t))
            
            buffer.extend(collected_trajectories)
            obs, info = env.reset()
            episode_traj = {0: [], 1: [], 2: []}
            
            # Ensure cumulative trackers exist
            if 'cumulative_mode_eps' not in locals():
                cumulative_mode_eps = {"normal": 0, "betli": 0, "durchmars": 0, "ulti": 0}
                cumulative_mode_wins = {"normal": 0, "betli": 0, "durchmars": 0, "ulti": 0}
            
            if len(buffer.obs) >= params["update_frequency"]:
                # Optimization step (PPO)
                agent.train()
                a_loss, c_loss, e_loss = update_agent(agent, optimizer, buffer, params)
                buffer.reset()
                
                writer.add_scalar("Metrics/CurriculumPhase", phase, global_step)
                torch.save(agent.state_dict(), 'models/agent_checkpoint.pth')
                
                win_rate = batch_wins / max(1, batch_episodes)
                recent_win_rates.append(win_rate)
                if len(recent_win_rates) > 10:
                    recent_win_rates.pop(0)
                    
                smoothed_wr = sum(recent_win_rates) / len(recent_win_rates)
                if smoothed_wr > 0.85 and len(recent_win_rates) == 10:
                    if phase < 3:
                        phase += 1
                        recent_win_rates.clear()
                        print(f"\n*** PROGRESSED TO CURRICULUM PHASE {phase} ***\n")
                        if phase == 3:
                            env = UltiEnv(curriculum_mode=False)
                
                if time.time() - last_print > 2:
                    print(f"Phase {phase} | Step: {global_step} | Win Rate: {win_rate:.2f} | Smoothed: {smoothed_wr:.2f} | Entropy: {e_loss:.4f}")
                    last_print = time.time()
                
                writer.add_scalar("Loss/Actor", a_loss, global_step)
                writer.add_scalar("Loss/Critic", c_loss, global_step)
                writer.add_scalar("Loss/Entropy", e_loss, global_step)
                
                writer.add_scalar("Metrics/WinRate_Overall", win_rate, global_step)
                writer.add_scalar("Metrics/Reward_Overall", batch_rewards / max(1, batch_episodes), global_step)
            
            if 'cumulative_mode_eps' not in locals():
                cumulative_mode_eps = {}
                cumulative_mode_wins = {}
            
            # Continuously update the trackers every step so we have real-time data
            for m in mode_eps.keys():
                if m not in cumulative_mode_eps:
                    cumulative_mode_eps[m] = 0
                    cumulative_mode_wins[m] = 0
                if mode_eps[m] > 0:
                    batch_wr = mode_wins[m] / mode_eps[m]
                    writer.add_scalar(f"Metrics/WinRate_{m.replace(' ', '_')}", batch_wr, global_step)
                    
                    cumulative_mode_eps[m] += mode_eps[m]
                    cumulative_mode_wins[m] += mode_wins[m]
                    mode_eps[m] = 0  # Reset so we don't double count
                    mode_wins[m] = 0
            
            # Dump JSON fast (every 500 steps)
            if global_step % 500 == 0:
                percentages = {}
                win_rates = {}
                total_cumulative_eps = max(1, sum(cumulative_mode_eps.values()))
                
                for m in cumulative_mode_eps.keys():
                    percentages[m] = (cumulative_mode_eps[m] / total_cumulative_eps) * 100
                    if cumulative_mode_eps[m] > 0:
                        win_rates[m] = (cumulative_mode_wins[m] / cumulative_mode_eps[m]) * 100
                    else:
                        win_rates[m] = 0.0
                        
                import json
                with open("logs/bidding_percentages.json", "w") as f:
                    json.dump({
                        "percentages": percentages, 
                        "win_rates": win_rates,
                        "totals": cumulative_mode_eps
                    }, f)
                    
            batch_episodes = 0
            batch_wins = 0
            batch_rewards = 0.0
            mode_wins.clear()

if __name__ == "__main__":
    train()
