import torch
from model import GPTLanguageModel
import json

config = json.load(open('../models/config.json'))
hparams = config['hparams']

print(hparams['device'])

# all the unique characters that occur in the text
chars = config['chars']
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

model = GPTLanguageModel(hparams['vocab_size'], hparams['n_emb_d'], hparams['n_heads'], hparams['n_layers'], hparams['dropout'], hparams['block_size'], hparams['device']).to(hparams['device'])
model.load_state_dict(torch.load('../models/gpt.pt'))
model.eval()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=hparams['device'])
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))