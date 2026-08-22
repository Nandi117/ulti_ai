import re

file_path = r'C:\ulti_ai\train.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix first indent
content = content.replace('        opt, is_rigged_episode = get_rigged_options(total_eps)\n    obs, info = env.reset(options=opt)', '    opt, is_rigged_episode = get_rigged_options(total_eps)\n    obs, info = env.reset(options=opt)')

# Fix second indent
content = content.replace('                        opt, is_rigged_episode = get_rigged_options(total_eps)\n            obs, info = env.reset(options=opt)', '            opt, is_rigged_episode = get_rigged_options(total_eps)\n            obs, info = env.reset(options=opt)')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
