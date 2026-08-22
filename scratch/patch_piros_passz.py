import re

file_path = r'C:\ulti_ai\train.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('forced_bid_id = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])', 'forced_bid_id = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
