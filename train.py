import os
import time
import json
import torch
import torch.optim as optim
import torch.nn as nn
from collections import defaultdict, deque
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

from engine.environments.ulti import UltiEnv
from agent.ppo import PPOMultiHeadAgent, ParallelReplayBuffer
from agent.baselines.heuristic import HeuristicAgent

def load_hyperparams(path):
    with open(path, "r") as f:
        return json.load(f)

def update_agent(agent, optimizer, buffer, params, writer, global_step, prefix="Agent"):
    if len(buffer) == 0:
        return 0, 0, 0
    
    device = next(agent.parameters()).device
    
    b_obs_dicts, b_actions, b_logprobs, b_rewards, b_values, b_dones, b_masks, b_modes = buffer.get_all()
    
    b_rewards = torch.tensor(b_rewards, dtype=torch.float32, device=device)
    b_values = torch.tensor(b_values, dtype=torch.float32, device=device)
    b_dones = torch.tensor(b_dones, dtype=torch.float32, device=device)
    
    # Calculate returns and advantages
    with torch.no_grad():
        b_returns = torch.zeros_like(b_rewards)
        b_advantages = torch.zeros_like(b_rewards)
        last_gae_lam = 0
        for t in reversed(range(len(b_rewards))):
            if t == len(b_rewards) - 1:
                next_non_terminal = 1.0 - b_dones[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - b_dones[t]
                next_value = b_values[t+1]
                
            delta = b_rewards[t] + params["gamma"] * next_value * next_non_terminal - b_values[t]
            b_advantages[t] = last_gae_lam = delta + params["gamma"] * params["gae_lambda"] * next_non_terminal * last_gae_lam
        b_returns = b_advantages + b_values
        
    # Flatten
    b_actions = torch.tensor(b_actions, dtype=torch.long, device=device)
    b_logprobs = torch.tensor(b_logprobs, dtype=torch.float32, device=device)
    b_advantages = b_advantages.to(device)
    b_returns = b_returns.to(device)
    b_values = b_values.to(device)
    
    # Normalize advantages
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    
    # Prepare Dict of Tensors for observation
    b_obs_merged = {
        "hand": [], "trick_history": [], "deduction_flags": [], 
        "trump_suit": [], "lead_suit": [], "scores": [], "belief_state": []
    }
    
    for obs in b_obs_dicts:
        b_obs_merged["hand"].append(obs["hand"])
        b_obs_merged["trick_history"].append(obs["trick_history"])
        b_obs_merged["deduction_flags"].append(obs["deduction_flags"])
        b_obs_merged["trump_suit"].append(obs["trump_suit"])
        b_obs_merged["lead_suit"].append(obs["lead_suit"])
        b_obs_merged["scores"].append(obs["scores"])
        b_obs_merged["belief_state"].append(obs["belief_state"])
        
    for k in b_obs_merged.keys():
        b_obs_merged[k] = torch.tensor(np.array(b_obs_merged[k]), dtype=torch.float32, device=device)
        
    b_masks = torch.tensor(np.array(b_masks), dtype=torch.bool, device=device)
    
    # Optimizing
    clip_fracs = []
    total_a_loss = 0
    total_c_loss = 0
    total_e_loss = 0
    
    for epoch in range(params["ppo_epochs"]):
        # Since we use mode switches, we must process sample by sample or group by mode.
        # For simplicity, we can do a forward pass sample by sample in a loop, or just pass normal.
        # Grouping by mode is much faster.
        
        mode_indices = defaultdict(list)
        for i, m in enumerate(b_modes):
            mode_indices[m].append(i)
            
        a_loss = torch.tensor(0.0, device=device)
        c_loss = torch.tensor(0.0, device=device)
        e_loss = torch.tensor(0.0, device=device)
        
        for m, indices in mode_indices.items():
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            
            m_obs = {k: v[idx_tensor] for k, v in b_obs_merged.items()}
            m_masks = b_masks[idx_tensor]
            m_actions = b_actions[idx_tensor]
            
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                m_obs, mode=m, action_mask=m_masks, action=m_actions
            )
            
            m_logprobs = b_logprobs[idx_tensor]
            m_advantages = b_advantages[idx_tensor]
            m_returns = b_returns[idx_tensor]
            m_values = b_values[idx_tensor]
            
            logratio = newlogprob - m_logprobs
            ratio = logratio.exp()
            
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_fracs += [((ratio - 1.0).abs() > params["clip_ratio"]).float().mean().item()]
                
            mb_advantages = m_advantages
            
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - params["clip_ratio"], 1 + params["clip_ratio"])
            m_a_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            newvalue = newvalue.view(-1)
            v_loss_unclipped = (newvalue - m_returns) ** 2
            v_clipped = m_values + torch.clamp(
                newvalue - m_values,
                -params["clip_ratio"],
                params["clip_ratio"],
            )
            v_loss_clipped = (v_clipped - m_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            m_c_loss = 0.5 * v_loss_max.mean()
            
            m_e_loss = entropy.mean()
            
            weight = len(indices) / len(b_modes)
            a_loss += m_a_loss * weight
            c_loss += m_c_loss * weight
            e_loss += m_e_loss * weight
            
        loss = a_loss - params["entropy_coef"] * e_loss + c_loss * 0.5
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        
        total_a_loss += a_loss.item()
        total_c_loss += c_loss.item()
        total_e_loss += e_loss.item()
        
    writer.add_scalar(f"Loss/{prefix}_Policy", total_a_loss / params["ppo_epochs"], global_step)
    writer.add_scalar(f"Loss/{prefix}_Value", total_c_loss / params["ppo_epochs"], global_step)
    writer.add_scalar(f"Loss/{prefix}_Entropy", total_e_loss / params["ppo_epochs"], global_step)
    
    return total_a_loss, total_c_loss, total_e_loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Multi-Phase Curriculum Training on {device} (Two-Brain Split Architecture)...")
    
    hyperparams_file = "config/hyperparams.json"
    params = load_hyperparams(hyperparams_file)
    
    declarer_agent = PPOMultiHeadAgent().to(device)
    defender_agent = PPOMultiHeadAgent().to(device)
    
    checkpoint_path = r'C:\ulti_ai\models\agent_checkpoint_split.pth'
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            declarer_agent.load_state_dict(state_dict['declarer'])
            defender_agent.load_state_dict(state_dict['defender'])
            print("Successfully loaded pre-trained split-brain checkpoint!")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
    else:
        print("No checkpoint found. Starting from scratch.")
        
    opt_decl = optim.Adam(declarer_agent.parameters(), lr=params["learning_rate"], eps=1e-5)
    opt_def = optim.Adam(defender_agent.parameters(), lr=params["learning_rate"], eps=1e-5)
    
    run_name = f"split_brain_{int(time.time())}"
    writer = SummaryWriter(f"logs/tb/{run_name}")
    print(f"Tensorboard logs saving to logs/tb/{run_name}")
    
    buffer_decl = ParallelReplayBuffer()
    buffer_def = ParallelReplayBuffer()
    
    heuristic = HeuristicAgent()
    env = UltiEnv(curriculum_mode=False, training_filter_mode=True)
    
    global_step = 0
    phase = 3
    
    mode_eps = defaultdict(int)
    mode_wins = defaultdict(int)
    cumulative_mode_eps = defaultdict(int)
    cumulative_mode_wins = defaultdict(int)
    recent_games = deque(maxlen=2000)
    
    obs, info = env.reset()
    episode_traj_decl = {0: [], 1: [], 2: []}
    episode_traj_def = {0: [], 1: [], 2: []}
    
    last_print = time.time()
    
    while True:
        current_player = env.current_player
        is_declarer = 1.0 if env.auction.highest_bidder == current_player else 0.0
        
        # Bidding uses Declarer Brain, playing as declarer uses Declarer Brain.
        # Playing as defender uses Defender Brain.
        
        if env.phase == "drop_talon":
            dash_mode = "Talon"
            nn_mode = "talon"
            active_agent = declarer_agent
        elif env.phase == "decision_to_rob" or env.phase == "bidding":
            dash_mode = "Robbing" if env.phase == "decision_to_rob" else "Bidding"
            nn_mode = "decision_to_rob" if env.phase == "decision_to_rob" else "normal"
            active_agent = declarer_agent
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
            
            active_agent = declarer_agent if is_declarer == 1.0 else defender_agent
            
        if dash_mode not in mode_eps:
            mode_eps[dash_mode] = 0
        if dash_mode not in mode_wins:
            mode_wins[dash_mode] = 0
            
        mask = info["action_mask"]
        legal_actions = [i for i, m in enumerate(mask) if m]
        
        with torch.no_grad():
            action, logprob, entropy, value = active_agent.get_action_and_value(
                obs, mode=nn_mode, action_mask=torch.tensor(mask, device=device)
            )
            action_item = action.item()
            logprob_item = logprob.item()
            value_item = value.item()
        
        next_obs, reward, terminated, truncated, info = env.step(action_item)
        
        if active_agent == declarer_agent:
            episode_traj_decl[current_player].append((obs, action_item, logprob_item, value_item, mask, nn_mode))
        else:
            episode_traj_def[current_player].append((obs, action_item, logprob_item, value_item, mask, nn_mode))
            
        obs = next_obs
        global_step += 1
        
        if terminated or truncated:
            if dash_mode not in cumulative_mode_eps:
                cumulative_mode_eps[dash_mode] = 0
                cumulative_mode_wins[dash_mode] = 0
                
            mode_eps[dash_mode] += 1
            cumulative_mode_eps[dash_mode] += 1
            if reward > 0:
                mode_wins[dash_mode] += 1
                cumulative_mode_wins[dash_mode] += 1
                recent_games.append((dash_mode, True))
            else:
                recent_games.append((dash_mode, False))
                
            # Process Declarer trajectories
            for p_id, traj in episode_traj_decl.items():
                for i, transition in enumerate(traj):
                    obs_t, action_t, logprob_t, value_t, mask_t, mode_t = transition
                    if i == len(traj) - 1:
                        step_reward = reward if env.auction.highest_bidder == p_id else -reward
                        step_done = True
                    else:
                        step_reward = 0.0
                        step_done = False
                    buffer_decl.push(obs_t, action_t, logprob_t, step_reward, value_t, step_done, mask_t, mode_t)
            
            # Process Defender trajectories
            for p_id, traj in episode_traj_def.items():
                for i, transition in enumerate(traj):
                    obs_t, action_t, logprob_t, value_t, mask_t, mode_t = transition
                    if i == len(traj) - 1:
                        step_reward = -reward # Since they are defenders, their reward is inverted
                        step_done = True
                    else:
                        step_reward = 0.0
                        step_done = False
                    buffer_def.push(obs_t, action_t, logprob_t, step_reward, value_t, step_done, mask_t, mode_t)
            
            # Reset
            episode_traj_decl = {0: [], 1: [], 2: []}
            episode_traj_def = {0: [], 1: [], 2: []}
            obs, info = env.reset()
            
            # Update Agents
            if len(buffer_decl) >= params["update_frequency"]:
                update_agent(declarer_agent, opt_decl, buffer_decl, params, writer, global_step, prefix="Declarer")
                buffer_decl.reset()
                
                # Save checkpoint (we can save both into the same dict, or separate files)
                torch.save({
                    'declarer': declarer_agent.state_dict(),
                    'defender': defender_agent.state_dict()
                }, r'C:\ulti_ai\models\agent_checkpoint_split.pth')
                
            if len(buffer_def) >= params["update_frequency"]:
                update_agent(defender_agent, opt_def, buffer_def, params, writer, global_step, prefix="Defender")
                buffer_def.reset()
                
            # Dump JSON dashboard stats
            if not hasattr(env, 'last_json_dump_episodes'):
                env.last_json_dump_episodes = 0
            
            total_eps = sum(cumulative_mode_eps.values())
            if total_eps - env.last_json_dump_episodes > 100:
                env.last_json_dump_episodes = total_eps
                percentages = {}
                win_rates = {}
                
                for m in cumulative_mode_eps.keys():
                    percentages[m] = (cumulative_mode_eps[m] / max(1, total_eps)) * 100
                    if cumulative_mode_eps[m] > 0:
                        win_rates[m] = (cumulative_mode_wins[m] / cumulative_mode_eps[m]) * 100
                    else:
                        win_rates[m] = 0.0
                        
                recent_eps = defaultdict(int)
                recent_wins = defaultdict(int)
                for mode, won in recent_games:
                    recent_eps[mode] += 1
                    if won:
                        recent_wins[mode] += 1
                        
                recent_percentages = {}
                recent_win_rates = {}
                total_recent = len(recent_games)
                for m in recent_eps.keys():
                    recent_percentages[m] = (recent_eps[m] / max(1, total_recent)) * 100
                    recent_win_rates[m] = (recent_wins[m] / recent_eps[m]) * 100
                        
                with open("logs/bidding_percentages.json", "w") as f:
                    json.dump({
                        "percentages": percentages, 
                        "win_rates": win_rates,
                        "totals": cumulative_mode_eps,
                        "recent_percentages": recent_percentages,
                        "recent_win_rates": recent_win_rates,
                        "recent_totals": recent_eps
                    }, f)
                    
            if time.time() - last_print > 5:
                print(f"Step: {global_step} | Total Episodes: {total_eps}")
                for m in mode_eps.keys():
                    if mode_eps[m] > 0:
                        print(f"  {m} | WR: {mode_wins[m]/mode_eps[m]:.2f}")
                print("-" * 30)
                mode_eps.clear()
                mode_wins.clear()
                last_print = time.time()

if __name__ == "__main__":
    train()
