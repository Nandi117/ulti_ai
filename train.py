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
from agent.league import League

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
        "trump_suit": [], "lead_suit": [], "scores": [], 
        "belief_state": [], "public_belief_state": [], "talon_first_drop": []
    }
    
    for obs in b_obs_dicts:
        b_obs_merged["hand"].append(obs["hand"])
        b_obs_merged["trick_history"].append(obs["trick_history"])
        b_obs_merged["deduction_flags"].append(obs["deduction_flags"])
        b_obs_merged["trump_suit"].append(obs["trump_suit"])
        b_obs_merged["lead_suit"].append(obs["lead_suit"])
        b_obs_merged["scores"].append(obs["scores"])
        b_obs_merged["belief_state"].append(obs["belief_state"])
        b_obs_merged["public_belief_state"].append(obs["public_belief_state"])
        b_obs_merged["talon_first_drop"].append(obs["talon_first_drop"])
        
    for k in b_obs_merged.keys():
        if k == "talon_first_drop":
            b_obs_merged[k] = torch.tensor(np.array(b_obs_merged[k]), dtype=torch.long, device=device)
        else:
            b_obs_merged[k] = torch.tensor(np.array(b_obs_merged[k]), dtype=torch.float32, device=device)
        
    b_masks = torch.tensor(np.array(b_masks), dtype=torch.bool, device=device)
    
    # Optimizing
    clip_fracs = []
    total_a_loss = 0
    total_c_loss = 0
    total_e_loss = 0
    
    for epoch in range(params["ppo_epochs"]):
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


def compute_bid_bonus(total_eps):
    """
    Gold Rush mechanic removed! 
    The agent now trains on true unadulterated points because it was pre-trained via Behavioral Cloning.
    """
    return 0.0


def get_rigged_options(total_eps):
    import random
    if total_eps < 300_000:
        prob = 1.0
    elif total_eps < 500_000:
        prob = 1.0 - ((total_eps - 300_000) / 200_000)
    else:
        prob = 0.0
        
    if random.random() < prob:
        forced_bid_id = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        return {"forced_bid_id": forced_bid_id}, True
    return None, False

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting v2 Training: Public Belief + Autoregressive Talon + League + Reward Shaping on {device}...")
    
    hyperparams_file = "config/hyperparams.json"
    params = load_hyperparams(hyperparams_file)
    
    declarer_agent = PPOMultiHeadAgent().to(device)
    defender_agent = PPOMultiHeadAgent().to(device)
    
    # Try loading checkpoint (will fail on obs_dim mismatch for old models — that's expected)
    checkpoint_path = r'C:\ulti_ai\models\agent_checkpoint_split.pth'
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            declarer_agent.load_state_dict(state_dict['declarer'])
            defender_agent.load_state_dict(state_dict['defender'])
            print("Successfully loaded pre-trained split-brain checkpoint!")
        except Exception as e:
            print(f"Could not load checkpoint (expected for new architecture): {e}")
            print("Starting fresh training with v2 architecture.")
    else:
        print("No checkpoint found. Starting fresh training with v2 architecture.")
        
    opt_decl = optim.Adam(declarer_agent.parameters(), lr=params["learning_rate"], eps=1e-5)
    
    # Asymmetric learning: Defender learns 10x slower so Declarer has room to experiment
    defender_lr = params["learning_rate"] / 10.0
    opt_def = optim.Adam(defender_agent.parameters(), lr=defender_lr, eps=1e-5)
    
    run_name = f"v2_league_{int(time.time())}"
    writer = SummaryWriter(f"logs/tb/{run_name}")
    print(f"Tensorboard logs saving to logs/tb/{run_name}")
    
    buffer_decl = ParallelReplayBuffer()
    buffer_def = ParallelReplayBuffer()
    
    # League Training: pool of frozen past selves
    league = League(max_snapshots=20)
    league_path = r'C:\ulti_ai\models\league.pth'
    league.load(league_path)
    league_prob = params.get("fictitious_play_prob", 0.2)
    league_opponent_decl = None
    league_opponent_def = None
    use_league_this_episode = False
    
    env = UltiEnv(curriculum_mode=False, training_filter_mode=True)
    
    global_step = 0
    total_eps = 0
    
    mode_eps = defaultdict(int)
    mode_wins = defaultdict(int)
    cumulative_mode_eps = defaultdict(int)
    cumulative_mode_wins = defaultdict(int)
    recent_games = deque(maxlen=2000)
    
    # Reward tracking for TensorBoard
    declarer_rewards_window = deque(maxlen=100)
    defender_rewards_window = deque(maxlen=100)
    
    # League snapshot interval
    league_snapshot_interval = 5000
    last_league_snapshot_eps = 0
    
    opt, is_rigged_episode = get_rigged_options(total_eps)
    obs, info = env.reset(options=opt)
    episode_traj_decl = {0: [], 1: [], 2: []}
    episode_traj_def = {0: [], 1: [], 2: []}
    
    # Decide if this first episode uses league opponents
    use_league_this_episode = False  # No snapshots yet
    
    last_print = time.time()
    
    while True:
        current_player = env.current_player
        is_declarer = 1.0 if env.auction.highest_bidder == current_player else 0.0
        
        # Determine mode and active agent
        if env.phase == "drop_talon":
            dash_mode = "Talon"
            # Autoregressive talon: use "talon" for first card, "talon_2" for second
            if env.talon_first_drop == 32:
                nn_mode = "talon"
            else:
                nn_mode = "talon_2"
            active_agent = declarer_agent
            is_league_player = False
        elif env.phase == "decision_to_rob" or env.phase == "bidding":
            dash_mode = "Robbing" if env.phase == "decision_to_rob" else "Bidding"
            nn_mode = "decision_to_rob" if env.phase == "decision_to_rob" else "normal"
            active_agent = declarer_agent
            is_league_player = False
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
            
            # League training: for opponent players, possibly use a frozen past agent
            if use_league_this_episode and current_player != 0 and league_opponent_decl is not None:
                active_agent = league_opponent_decl if is_declarer == 1.0 else league_opponent_def
                is_league_player = True
            else:
                active_agent = declarer_agent if is_declarer == 1.0 else defender_agent
                is_league_player = False
            
        if dash_mode not in mode_eps:
            mode_eps[dash_mode] = 0
        if dash_mode not in mode_wins:
            mode_wins[dash_mode] = 0
            
        mask = info["action_mask"]
        
        is_bidding = (env.phase == "bidding")
        
        with torch.no_grad():
            action, logprob, entropy, value = active_agent.get_action_and_value(
                obs, mode=nn_mode, action_mask=torch.tensor(mask, device=device)
            )
            action_item = action.item()
            
            if is_bidding and is_rigged_episode:
                if hasattr(env, 'forced_bid_id') and env.forced_bid_id is not None:
                    if current_player == 0:
                        if mask[env.forced_bid_id]:
                            action_item = env.forced_bid_id
                    else:
                        if mask[0]: # Opponents must pass so P0 can practice the rigged hand
                            action_item = 0
                    
            logprob_item = logprob.item()
            value_item = value.item()
        
        next_obs, reward, terminated, truncated, info = env.step(action_item)
        
        # Only store trajectories for the LEARNING agents (not league opponents)
        if not is_league_player:
            skip_store = (is_bidding and is_rigged_episode)
            if not skip_store:
                if active_agent == declarer_agent:
                    episode_traj_decl[current_player].append((obs, action_item, logprob_item, value_item, mask, nn_mode))
                else:
                    episode_traj_def[current_player].append((obs, action_item, logprob_item, value_item, mask, nn_mode))
            
        obs = next_obs
        global_step += 1
        
        if terminated or truncated:
            total_eps += 1
            
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
            
            # === REWARD SHAPING ===
            bid_bonus = compute_bid_bonus(total_eps)
            bid = env.auction.highest_bid
            bonus = 0.0
            if bid is not None and bid.id != 0:
                # Supercharged bonus: Flat multiplier on the bid's points
                # E.g. Ulti (4 pts) * 1.5 bonus = +6.0 reward just for bidding!
                bonus = bid_bonus * bid.points
            
            # === Process Declarer trajectories ===
            ep_decl_reward = 0.0
            for p_id, traj in episode_traj_decl.items():
                for i, transition in enumerate(traj):
                    obs_t, action_t, logprob_t, value_t, mask_t, mode_t = transition
                    if i == len(traj) - 1:
                        base_reward = reward if env.auction.highest_bidder == p_id else -reward
                        # Apply bid bonus ONLY to the declarer (the one who bid)
                        if env.auction.highest_bidder == p_id and bid is not None and bid.id != 0:
                            step_reward = base_reward + bonus
                        else:
                            step_reward = base_reward
                        step_done = True
                        ep_decl_reward = step_reward
                    else:
                        step_reward = 0.0
                        step_done = False
                    buffer_decl.push(obs_t, action_t, logprob_t, step_reward, value_t, step_done, mask_t, mode_t)
            
            # === Process Defender trajectories ===
            ep_def_reward = 0.0
            for p_id, traj in episode_traj_def.items():
                for i, transition in enumerate(traj):
                    obs_t, action_t, logprob_t, value_t, mask_t, mode_t = transition
                    if i == len(traj) - 1:
                        step_reward = -reward  # Defenders: inverted reward
                        step_done = True
                        ep_def_reward = step_reward
                    else:
                        step_reward = 0.0
                        step_done = False
                    buffer_def.push(obs_t, action_t, logprob_t, step_reward, value_t, step_done, mask_t, mode_t)
            
            # Track rewards for TensorBoard
            declarer_rewards_window.append(ep_decl_reward)
            defender_rewards_window.append(ep_def_reward)
            
            # === TensorBoard Reward Curves ===
            if total_eps % 100 == 0:
                avg_decl = sum(declarer_rewards_window) / max(1, len(declarer_rewards_window))
                avg_def = sum(defender_rewards_window) / max(1, len(defender_rewards_window))
                writer.add_scalar("Reward/Declarer_Avg100", avg_decl, total_eps)
                writer.add_scalar("Reward/Defender_Avg100", avg_def, total_eps)
                writer.add_scalar("Curriculum/BidBonus", bid_bonus, total_eps)
                writer.add_scalar("League/SnapshotCount", len(league), total_eps)
            
            # Reset episode
            episode_traj_decl = {0: [], 1: [], 2: []}
            episode_traj_def = {0: [], 1: [], 2: []}
            
            opt, is_rigged_episode = get_rigged_options(total_eps)
            obs, info = env.reset(options=opt)
            
            # === League: decide if next episode uses league opponents ===
            if league.has_opponents() and random.random() < league_prob:
                use_league_this_episode = True
                league_opponent_decl, league_opponent_def = league.sample_opponent(device)
            else:
                use_league_this_episode = False
                league_opponent_decl = None
                league_opponent_def = None
            
            # === League: save snapshot periodically ===
            # === League Saving ===
            if total_eps > 0 and total_eps % 50_000 == 0:
                league.add_snapshot(declarer_agent, defender_agent)
                league.save(r'C:\ulti_ai\models\league.pth')
                last_league_snapshot_eps = total_eps
                print(f"[League] Saved snapshot #{len(league)} at episode {total_eps}")
            
            # Update Agents
            if len(buffer_decl) >= params["update_frequency"]:
                update_agent(declarer_agent, opt_decl, buffer_decl, params, writer, global_step, prefix="Declarer")
                buffer_decl.reset()
                
                # Save checkpoint
                torch.save({
                    'declarer': declarer_agent.state_dict(),
                    'defender': defender_agent.state_dict()
                }, r'C:\ulti_ai\models\agent_checkpoint_split.pth')
                
            if len(buffer_def) >= params["update_frequency"]:
                # Phase 1 Curriculum: Freeze defender completely for first 50k games
                if total_eps >= 50_000:
                    update_agent(defender_agent, opt_def, buffer_def, params, writer, global_step, prefix="Defender")
                buffer_def.reset()
                
            # Dump JSON dashboard stats
            if not hasattr(env, 'last_json_dump_episodes'):
                env.last_json_dump_episodes = 0
            
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
                        
                recent_eps_dict = defaultdict(int)
                recent_wins_dict = defaultdict(int)
                for mode, won in recent_games:
                    recent_eps_dict[mode] += 1
                    if won:
                        recent_wins_dict[mode] += 1
                        
                recent_percentages = {}
                recent_win_rates = {}
                total_recent = len(recent_games)
                for m in recent_eps_dict.keys():
                    recent_percentages[m] = (recent_eps_dict[m] / max(1, total_recent)) * 100
                    recent_win_rates[m] = (recent_wins_dict[m] / recent_eps_dict[m]) * 100
                        
                with open("logs/bidding_percentages.json", "w") as f:
                    json.dump({
                        "percentages": percentages, 
                        "win_rates": win_rates,
                        "totals": cumulative_mode_eps,
                        "recent_percentages": recent_percentages,
                        "recent_win_rates": recent_win_rates,
                        "recent_totals": recent_eps_dict
                    }, f)
                    
            if time.time() - last_print > 5:
                print(f"Step: {global_step} | Eps: {total_eps} | Bid Bonus: {compute_bid_bonus(total_eps):.2f} | League: {len(league)} snapshots")
                for m in mode_eps.keys():
                    if mode_eps[m] > 0:
                        print(f"  {m} | WR: {mode_wins[m]/mode_eps[m]:.2f}")
                print("-" * 30)
                mode_eps.clear()
                mode_wins.clear()
                last_print = time.time()

if __name__ == "__main__":
    train()
