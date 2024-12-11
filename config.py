from dataclasses import dataclass
import torch

@dataclass
class parameters:

    # Hyperparameters
    max_iters = 100 # 10
    eval_interval = 50 # 50
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
    learning_rate = 1e-4