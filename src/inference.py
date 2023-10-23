import torch
from model import GPTLanguageModel
import json

CONFIG_PATH='../models/config.json'
MODEL_PATH = '../models/gpt.pt'

config = json.load(open(CONFIG_PATH))
hparams = config['hparams']

# all the unique characters that occur in the text
chars = config['chars']
vocab_size = len(chars)

# START --- taken from src/train_model.py ---
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
# END --- taken from src/train_model.py ---

model = GPTLanguageModel(hparams['vocab_size'], hparams['n_emb_d'], hparams['n_heads'], hparams['n_layers'], hparams['dropout'], hparams['block_size'], hparams['device']).to(hparams['device'])
model.load_state_dict(torch.load('../models/gpt.pt'))
model.eval()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=hparams['device']) # newline
context = torch.tensor(encode('Sokrate liess den Wald mit den Werwölfen hinter sich und '), dtype=torch.long, device=hparams['device']).unsqueeze(0) # custom context

print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))