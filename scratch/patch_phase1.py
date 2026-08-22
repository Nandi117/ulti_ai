import re
file_path = r'C:\ulti_ai\phase1_supervised.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("torch.save(agent.state_dict(), r'C:\ulti_ai\models\agent_checkpoint_split.pth')", "torch.save({'declarer': agent.state_dict(), 'defender': agent.state_dict()}, r'C:\ulti_ai\models\agent_checkpoint_split.pth')")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
