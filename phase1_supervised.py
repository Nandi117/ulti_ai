import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from engine.environments.ulti import UltiEnv
from engine.heuristic_bidder import OracleBidder
from agent.ppo import PPOMultiHeadAgent
from engine.trick import ALL_CARDS
import os

def bin_to_cards(bin_hand):
    return [ALL_CARDS[i] for i, val in enumerate(bin_hand) if val == 1]

def run_phase_1():
    print("Starting Phase 1: Behavioral Cloning (The Bidding Exam)...")
    env = UltiEnv()
    agent = PPOMultiHeadAgent()
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    
    losses = []
    for i in range(10000):
        obs, _ = env.reset()
        
        p0 = bin_to_cards(env.hands[0])
        p1 = bin_to_cards(env.hands[1])
        p2 = bin_to_cards(env.hands[2])
        
        target_bid = OracleBidder.get_best_bid(p0, p1, p2)
        
        obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in obs.items()}
        probs, _ = agent(obs_tensor)
        
        loss = -torch.log(probs[0, target_bid] + 1e-8)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if i % 1000 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            print(f"Step {i:05d}, Avg Loss: {avg_loss:.4f}, Target Bid: {target_bid}")
            
    os.makedirs(r'C:\ulti_ai\models', exist_ok=True)
    checkpoint = {
        'declarer': agent.state_dict(),
        'defender': agent.state_dict()
    }
    torch.save(checkpoint, r'C:\ulti_ai\models\agent_checkpoint_split.pth')
    print("Phase 1 Complete! The Bidding Head has cloned the Oracle. Model saved.")

if __name__ == '__main__':
    run_phase_1()
