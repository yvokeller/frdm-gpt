import torch

from gpt import GPTLanguageModel

model = GPTLanguageModel()
model.load_state_dict(torch.load('models/gpt-4.pt'))
model.eval()