import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from .gpt_utils import Block

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()

    ''' Initialize Layers: Embedding Layers, Dropout, Block, LayerNorm, Linear '''
    self.embedding = nn.Embedding(config.vocab_size, config.embed_dim) # Token Embedding
    self.pos_embed = nn.Embedding(config.seq_len, config.embed_dim) # Positional Encoding
    self.drop = nn.Dropout(config.dropout) # Add dropout layer

    # Create Layer Blocks (with MultiHeadAttention)
    m = [] 
    for _ in range(config.num_layers):
      m.append(Block(config))
    self.blocks = nn.ModuleList(m) # Put blocks in module list

    self.ln = nn.LayerNorm(config.embed_dim) # Layer Norm Layer
    self.out = nn.Linear(config.embed_dim, config.vocab_size) # Linear Layer

    ''' Initialize configurations and scale parameters '''
    self.config = config # Initialize configuration
    self.apply(self._init_weights) # initialize the weights in each layer with a specific distribution and parameters

    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in self.named_parameters():
        if pn.endswith('out_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

  def forward(self, img_embeds, text, labels=None):

    # Make img_embeds in right processing device
    device = img_embeds.device

    ''' Text Embedding '''
    emb = self.embedding(text) # token embed text
    pos = torch.arange(0, self.config.seq_len, dtype=torch.long, device=device).unsqueeze(0) # Shape: (1, sequence_length)
    pos = self.pos_embed(pos) # position encode text
    x = self.drop(img_embeds + emb + pos) # Combines token embeddings with positional embeddings

    ''' Passing input through each block in the modelm and layernorm,linear '''
    for block in self.blocks:
      x = block(x)

    x = self.ln(x) # Layer norm
    x = self.out(x) # Linear
    
    ''' Calculating a loss if labels is passed (used for training) '''
    if labels is not None:
       return x, F.cross_entropy(x.view(-1, x.size(-1)), labels.view(-1), ignore_index=0) # Use Cross-entropy
    else:
       return x, None

  @torch.no_grad()
  def generate(self, image_embed, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    new = []
    for _ in range(max_new_tokens):
        
        ''' Generate predicted caption sequence'''
        
        # Cropping the last seq_len values in current sequence if it is greater than the sequence length
        idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]

        # Performancing inference
        logits, _ = self(image_embed, idx_cond)
        logits = logits[:, -1, :] / temperature # scale logits before softmax

        # Get top k items (not used atm)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Apply softmax
        probs = F.softmax(logits, dim=-1)
        
        # Multinomial sampling
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        idx = torch.cat((idx, idx_next), dim=1) # Concatenate
        new.append(idx_next)

    return idx, torch.tensor(new)
  
  def _init_weights(self, module):
        ''' Initialize Weights '''
        if isinstance(module, nn.Linear): # Linear weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): # Embedding weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  
  def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        ''' Configure optimizers, don't see this used though'''

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer