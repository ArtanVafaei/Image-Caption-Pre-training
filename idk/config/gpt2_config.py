from dataclasses import dataclass

@dataclass
class gpt2_config:
    num_layers = 3
    num_heads = 8
    seq_len = 128
    embed_dim = 1
    k_dim = None       # DO NOT CHANGE
    vocab_size = 50304  # DO NOT CHANGE: Size of the vocabulary that the model can understand
    dropout = 0.0
    is_casual = True    # DO NOT CHANGE
    
    # max_lr = 1e-4
    # min_lr = 1e-5
    

