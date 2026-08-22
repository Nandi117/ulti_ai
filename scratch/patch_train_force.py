import re

file_path = r'C:\ulti_ai\train.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

override_old = '''            if is_bidding and is_rigged_episode:
                from engine.heuristic_bidder import OracleBidder
                from phase1_supervised import bin_to_cards
                p0 = bin_to_cards(env.hands[0])
                p1 = bin_to_cards(env.hands[1])
                p2 = bin_to_cards(env.hands[2])
                oracle_bid = OracleBidder.get_best_bid(p0, p1, p2)
                if mask[oracle_bid]:
                    action_item = oracle_bid'''
                    
override_new = '''            if is_bidding and is_rigged_episode:
                if hasattr(env, 'forced_bid_id') and env.forced_bid_id is not None:
                    if mask[env.forced_bid_id]:
                        action_item = env.forced_bid_id'''

content = content.replace(override_old, override_new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
