import re

file_path = r'C:\ulti_ai\engine\environments\ulti.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add self.forced_bid_id = forced_bid_id
search_str = '''            if forced_bid_id is not None:
                import random'''
replace_str = '''            self.forced_bid_id = forced_bid_id
            if forced_bid_id is not None:
                import random'''
content = content.replace(search_str, replace_str)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
