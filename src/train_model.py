from model import GPTLanguageModel
import wandb
import time
import json
import torch

mode = 'test'

if mode == 'test':
    # TEST hparam config
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 8 # what is the maximum context length for predictions?
    max_iters = 1
    eval_interval = 50
    learning_rate = 1e-3
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    eval_iters = 200
    n_emb_d = 32 # number of embedding dimensions
    n_heads = 6
    n_layers = 6
    dropout = 0.2
elif mode == 'big':
    # BIG hparam config
    batch_size = 64 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4 # reduced as network is much bigge
    eval_iters = 200
    n_emb_d = 384 # number of embedding dimensions
    n_heads = 6
    n_layers = 6
    dropout = 0.2

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
dataset_path = '../data/ansturm.txt'

torch.manual_seed(42)

# load the text
with open(dataset_path, 'r', encoding='utf-8') as f:
    text = f.read()

    punctuation = {
        u'\u2003': " ",
    }
    
    for key, value in punctuation.items():
        text = text.replace(key, value)

# all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPTLanguageModel(vocab_size, n_emb_d, n_heads, n_layers, dropout, block_size, device)
model = model.to(device)

# WandB – Initialize a new run
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    entity="yvokeller",
    project="frdm-GPT",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "n_emb_d": n_emb_d,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
    })

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        wandb.log({"train_loss": losses['train'], "val_loss": losses['val']})

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save the model & config.json to disk with timestampf format YYYY-MM-DD_HH-MM-SS
torch.save(model.state_dict(), f'../models/gpt.pt')

with open("../models/config.json", "w") as f:
    f.write(json.dumps({
            'hparams': model.config,
            'chars': chars,
        }
    ))

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
