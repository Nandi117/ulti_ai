import torch
state = torch.load('C:/ulti_ai/models/agent_checkpoint.pth')
print("Keys:", state.keys())
print("policy_normal.weight shape:", state['policy_normal.weight'].shape)
print("policy_normal.bias shape:", state['policy_normal.bias'].shape)
