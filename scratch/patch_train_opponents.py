import re

file_path = r'C:\ulti_ai\train.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

override_old = '''            if is_bidding and is_rigged_episode:
                if hasattr(env, 'forced_bid_id') and env.forced_bid_id is not None:
                    if mask[env.forced_bid_id]:
                        action_item = env.forced_bid_id'''

override_new = '''            if is_bidding and is_rigged_episode:
                if hasattr(env, 'forced_bid_id') and env.forced_bid_id is not None:
                    if current_player == 0:
                        if mask[env.forced_bid_id]:
                            action_item = env.forced_bid_id
                    else:
                        if mask[0]: # Opponents must pass so P0 can practice the rigged hand
                            action_item = 0'''

content = content.replace(override_old, override_new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
