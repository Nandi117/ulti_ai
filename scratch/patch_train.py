import re

file_path = r'C:\ulti_ai\train.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add get_rigged_options function
rig_func = '''
def get_rigged_options(total_eps):
    import random
    if total_eps < 300_000:
        prob = 1.0
    elif total_eps < 500_000:
        prob = 1.0 - ((total_eps - 300_000) / 200_000)
    else:
        prob = 0.0
        
    if random.random() < prob:
        forced_bid_id = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
        return {"forced_bid_id": forced_bid_id}, True
    return None, False

'''
content = content.replace('def train():\n', rig_func + 'def train():\n')

# Replace first env.reset()
first_reset = '''    opt, is_rigged_episode = get_rigged_options(total_eps)
    obs, info = env.reset(options=opt)'''
content = re.sub(r'obs, info = env\.reset\(\)', first_reset, content, count=1)

# Replace second env.reset()
second_reset = '''            opt, is_rigged_episode = get_rigged_options(total_eps)
            obs, info = env.reset(options=opt)'''
content = re.sub(r'obs, info = env\.reset\(\)', second_reset, content)

# Update the Oracle Override logic
override_old = '''            if is_bidding and total_eps < 200_000:
                from engine.heuristic_bidder import OracleBidder
                from phase1_supervised import bin_to_cards
                p0 = bin_to_cards(env.hands[0])
                p1 = bin_to_cards(env.hands[1])
                p2 = bin_to_cards(env.hands[2])
                oracle_bid = OracleBidder.get_best_bid(p0, p1, p2)
                if mask[oracle_bid]:
                    action_item = oracle_bid
                    
            logprob_item = logprob.item()
            value_item = value.item()'''
override_new = '''            if is_bidding and is_rigged_episode:
                from engine.heuristic_bidder import OracleBidder
                from phase1_supervised import bin_to_cards
                p0 = bin_to_cards(env.hands[0])
                p1 = bin_to_cards(env.hands[1])
                p2 = bin_to_cards(env.hands[2])
                oracle_bid = OracleBidder.get_best_bid(p0, p1, p2)
                if mask[oracle_bid]:
                    action_item = oracle_bid
                    
            logprob_item = logprob.item()
            value_item = value.item()'''
content = content.replace(override_old, override_new)

# Update skip_store logic
skip_store_old = '''        # Only store trajectories for the LEARNING agents (not league opponents)
        if not is_league_player:
            skip_store = (is_bidding and total_eps < 200_000)'''
skip_store_new = '''        # Only store trajectories for the LEARNING agents (not league opponents)
        if not is_league_player:
            skip_store = (is_bidding and is_rigged_episode)'''
content = content.replace(skip_store_old, skip_store_new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched train.py successfully")
