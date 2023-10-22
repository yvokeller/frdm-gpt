import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ One head of self-attention (decoder block) """

    def __init__(self, n_emb_d, head_size, block_size, dropout):
        super().__init__()
        # note: typically no bias is used in self-attention
        self.key = nn.Linear(n_emb_d, head_size, bias=False)
        self.query = nn.Linear(n_emb_d, head_size, bias=False)
        self.value = nn.Linear(n_emb_d, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T) | note: scaling by sqrt of the head size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_emb_d, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_emb_d, head_size, block_size, dropout) for _ in range(num_heads)]) # instantiate num_heads heads
        self.proj = nn.Linear(head_size * num_heads, n_emb_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run the heads all in parallel into a list of tensors and concatenate the outputs over C (channel dimension)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) # linear transformation of the output from sa-heads for residual pathway
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_emb_d, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb_d, 4 * n_emb_d),
            nn.ReLU(),
            nn.Linear(4 * n_emb_d, n_emb_d), # projection layer for going back into the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_emb_d, num_heads, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_emb_d // num_heads # calculate the head size so that the total dimensionality channel-wise works out
        self.sa = MultiHeadAttention(n_emb_d, num_heads, head_size, block_size, dropout) # communication
        self.ffwd = FeedFoward(n_emb_d, dropout) # computation
        self.ln1 = nn.LayerNorm(n_emb_d)
        self.ln2 = nn.LayerNorm(n_emb_d)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # layer norm -> self-attention -> residual connection
        x = x + self.ffwd(self.ln2(x)) # layer norm -> feed forward -> residual connection
        return x

# GPT model
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_emb_d=384, n_heads=6, n_layers=6, dropout=0.2, block_size=256, device='mps'):
        super().__init__()

        # hparams
        self.vocab_size = vocab_size
        self.n_emb_d = n_emb_d
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.block_size = block_size
        self.device = device

        self.config = {
            'vocab_size': vocab_size,
            'n_emb_d': n_emb_d,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout': dropout,
            'block_size': block_size,
            'device': device,
        }

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb_d)
        # not only enbed token identity, but also position! each position from 0 to block_size-1 gets its own embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_emb_d)
        # blocks
        self.blocks = nn.Sequential(*[Block(n_emb_d, n_heads, block_size, dropout) for _ in range(n_layers)])
        # final layer norm
        self.ln_f = nn.LayerNorm(n_emb_d)
        # the final layer is a linear transformation that maps from n_emb_d dimensions to vocab_size logits
        self.lm_head = nn.Linear(n_emb_d, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C) | integer sequence from 0 to T-1
        x = token_emb + pos_emb # (B,T,C) | x now not only holds token identity, but also position at which the token occurs
        x = self.blocks(x) # (B,T,C) | note: this is run on a per-token level! self-attention is run on a per-sequence level and allowed the tokens to interact with each other
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
