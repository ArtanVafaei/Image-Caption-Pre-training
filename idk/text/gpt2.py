import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect

from .gpt_utils import Block

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
    self.pos_embed = nn.Embedding(config.seq_len, config.embed_dim)
    self.drop = nn.Dropout(config.dropout)

    m = []
    # creating the blocks
    for _ in range(config.num_layers):
      m.append(Block(config))
    
    self.blocks = nn.ModuleList(m)

    self.ln = nn.LayerNorm(config.embed_dim)
    self.out = nn.Linear(config.embed_dim, config.vocab_size)

    self.config = config

    # initialize the weights in each layer with a specific distribution and parameters
    self.apply(self._init_weights)

    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in self.named_parameters():
        if pn.endswith('out_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

  def forward(self, img_embeds, text, labels=None):
    device = img_embeds.device

    emb = self.embedding(text)
    pos = torch.arange(0, self.config.seq_len, dtype=torch.long, device=device).unsqueeze(0) # Shape: (1, sequence_length)
    pos = self.pos_embed(pos)

    # Combines token embeddings with positional embeddings
    x = self.drop(img_embeds + emb + pos)

    # passing input through each block in the model
    for block in self.blocks:
      x = block(x)

    x = self.ln(x)
    x = self.out(x)
    
    # calculating a loss if labels is passed (used for training)
    if labels is not None:
       return x, F.cross_entropy(x.view(-1, x.size(-1)), labels.view(-1), ignore_index=0)
    else:
       return x, None

  @torch.no_grad()
  def generate(self, image_embed, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    new = []
    for _ in range(max_new_tokens):
        # cropping the last seq_len values in current sequence if it is greater than the sequence length
        idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]

        # performancing inference on the 
        logits, _ = self(image_embed, idx_cond)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        idx = torch.cat((idx, idx_next), dim=1)
        new.append(idx_next)

    return idx, torch.tensor(new)
  
  def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  
  def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
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