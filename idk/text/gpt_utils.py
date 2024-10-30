import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    
    self.num_heads = config.num_heads
    self.embed_dim = config.embed_dim
    k_dim = config.embed_dim if config.k_dim is None else config.k_dim

    self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
    self.k_proj = nn.Linear(k_dim, config.embed_dim)
    self.v_proj = nn.Linear(k_dim, config.embed_dim)

    # Casual Self-Attention Mask
    # NOTE: Need to modify to make it more versitile for different types of masking or no masking at all depending on the task
    # self.register_buffer("mask", torch.tril(torch.ones(1, 1, config.seq_len, config.seq_len)))

    # output projection
    self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
  
  def forward(self, q, k, v, mask=None, is_casual=True, flash=False):
    # queries, keys, values will be shape (batch, sequence, channel)
    b, t, c = q.size()  

    # Calculating the queries, keys, values
    q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

    '''
    Splitting q, k, v across the heads --> (batch, sequence, num_heads, channel // num_heads)
    Then transposing to get shape (batch, num_heads, sequence, channel // num_heads)
    '''
    q = q.view(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)
    k = k.view(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)
    v = v.view(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)

    if flash:
       x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_casual)
    else:
      # (QK^T) / sqrt(channel)
      # This creates the attention matrix that is scaled to reduce its variance
      attn = (q @ k.transpose(-2, -1)) / math.sqrt(c)

      # applying a mask if there is one --> if we want to train the model to predict a part of the input sequence
    
      if is_casual:
        attn = attn.masked_fill(torch.tril(torch.ones(1, 1, t, t)).to(q.device) == 0, float("-inf"))
      elif mask is not None:
         assert attn.dim() == mask.dim()
         attn = attn.masked_fill(mask == 0, float("-inf"))

      # applying the softmax function to calculate probabilities 
      attn = F.softmax(attn, dim=-1)

      # using attention weights and multiplying it by the value
      x = attn @ v

    # going from shape (batch, num_heads, sequence, channel) to (batch, sequence, channel)
    x = x.transpose(1, 2).reshape(b, t, c)

    # output projection
    return self.out_proj(x)
  

'''
Contains the following:
LayerNorm: Normalizes the sequence 
MultiHeadAttention: The main crux of the model, allows it to learn the importance part of the input sequence with relation to each other
Another LayerNorm
A feedforward network that learns to extracts the relevant features from the MHA embeddings
'''
class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.embed_dim)
    self.attention = MultiHeadAttention(config)
    self.ln2 = nn.LayerNorm(config.embed_dim)
    self.mlp = nn.Sequential(
        nn.Linear(config.embed_dim, config.embed_dim * 4),
        nn.GELU(),
        nn.Linear(config.embed_dim * 4, config.embed_dim),
        nn.Dropout(config.dropout)
    )
    self.is_casual = config.is_casual

  def forward(self, x, attention_mask=None):
    y = self.ln1(x)
    x = x + self.attention(y, y, y, mask=attention_mask, is_casual=self.is_casual)
    x = x + self.mlp(self.ln2(x))
    return x