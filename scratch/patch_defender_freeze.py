import re

file_path = r'C:\ulti_ai\train.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the 50k defender freeze
old_logic = '''            if len(buffer_def) >= params["update_frequency"]:
                # Phase 1 Curriculum: Freeze defender completely for first 50k games
                if total_eps >= 50_000:
                    update_agent(defender_agent, opt_def, buffer_def, params, writer, global_step, prefix="Defender")
                buffer_def.reset()'''
                
new_logic = '''            if len(buffer_def) >= params["update_frequency"]:
                update_agent(defender_agent, opt_def, buffer_def, params, writer, global_step, prefix="Defender")
                buffer_def.reset()'''

content = content.replace(old_logic, new_logic)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
